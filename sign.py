#!/usr/bin/env python3
"""sign.py — Handwritten signature attribute analyser.

Scans a directory for PNG signature images, analyses each one using computer
vision (PIL/NumPy/SciPy) and an optional AI vision API (Anthropic or OpenAI),
then saves the resulting keyword list to a corresponding .txt file.
"""

__version__ = "1.0.0"

import argparse
import base64
import logging
import os
import pathlib
import re
import shutil
import sys
import time
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from rich.console import Console
from rich.theme import Theme

# ── Rich theme (from TUI_SPEC.md) ─────────────────────────────────────────────
_THEME = Theme({
    "info": "white",
    "detail": "bright_white",
    "success": "bright_green",
    "warning": "bright_yellow",
    "error": "bright_red",
    "header": "bright_magenta",
    "progress": "bright_blue",
    "muted": "dim white",
})

console = Console(theme=_THEME)

# ── Progress tracking ────────────────────────────────────────────────────────

# Module-level progress bar (set during main loop)
_progress = None
_progress_task = None

# ── Image discovery ───────────────────────────────────────────────────────────

def find_png_files(directory: str) -> list[pathlib.Path]:
    d = pathlib.Path(directory)
    files: set[pathlib.Path] = set(d.glob("*.png")) | set(d.glob("*.PNG"))
    return sorted(files)

# ── Computer-vision analysis helpers ─────────────────────────────────────────

def _kw_relative_size(bbox_h: int, bbox_w: int, canvas_h: int, canvas_w: int) -> str:
    ratio = (bbox_h * bbox_w) / (canvas_h * canvas_w)
    if ratio < 0.10:
        return "tiny"
    if ratio < 0.30:
        return "small"
    if ratio < 0.60:
        return "medium"
    return "large"


def _kw_aspect_ratio(bbox_h: int, bbox_w: int) -> str:
    ratio = bbox_w / bbox_h if bbox_h > 0 else 1.0
    if ratio < 0.75:
        return "tall"
    if ratio < 1.33:
        return "square"
    return "wide"


def _kw_ink_density(binary_crop: np.ndarray) -> str:
    density = float(binary_crop.mean())
    if density < 0.10:
        return "sparse-ink"
    if density < 0.30:
        return "light-ink"
    if density < 0.55:
        return "moderate-ink"
    return "dense-ink"


def _kw_stroke_width(dist: np.ndarray, binary: np.ndarray) -> str:
    ink_dists = dist[binary]
    if ink_dists.size == 0:
        return "thin-strokes"
    stroke_width = float(ink_dists.mean()) * 2
    if stroke_width < 3:
        return "hairline-strokes"
    if stroke_width < 7:
        return "thin-strokes"
    if stroke_width < 14:
        return "medium-strokes"
    return "thick-strokes"


def _kw_complexity(n_components: int) -> str:
    if n_components == 1:
        return "single-component"
    if n_components <= 5:
        return "few-components"
    if n_components <= 20:
        return "moderate-complexity"
    return "high-complexity"


def _kw_vertical_com(binary: np.ndarray, bbox_rmin: int, bbox_rmax: int) -> str:
    row_sums = binary.sum(axis=1)
    total = float(row_sums.sum())
    if total == 0:
        return "vertically-centered"
    row_indices = np.arange(binary.shape[0], dtype=float)
    com_row = float(np.average(row_indices, weights=row_sums))
    bbox_h = bbox_rmax - bbox_rmin
    if bbox_h == 0:
        return "vertically-centered"
    com_frac = (com_row - bbox_rmin) / bbox_h
    if com_frac < 0.40:
        return "top-heavy"
    if com_frac > 0.60:
        return "bottom-heavy"
    return "vertically-centered"


def _kw_height_span(bbox_h: int, canvas_h: int) -> str:
    ratio = bbox_h / canvas_h if canvas_h > 0 else 0.0
    if ratio < 0.20:
        return "compact-height"
    if ratio < 0.50:
        return "mid-height"
    if ratio < 0.75:
        return "tall-height"
    return "full-height"


def _kw_slant(binary: np.ndarray) -> str:
    rows, cols = np.where(binary)
    if len(rows) < 10:
        return "upright-slant"
    r = rows.astype(float)
    c = cols.astype(float)
    var_c = float(np.var(c))
    if var_c < 1e-6:
        return "upright-slant"
    cov_rc = float(np.cov(r, c)[0, 1])
    slope = cov_rc / var_c
    # slope < 0: row decreases as col increases → signature rises right → right-slanted
    # slope > 0: row increases as col increases → signature falls right → left-slanted
    angle_deg = float(np.degrees(np.arctan(slope)))
    if angle_deg < -15:
        return "right-slanted"
    if angle_deg > 15:
        return "left-slanted"
    return "upright-slant"


def _kw_connectivity(n_components: int, ink_pixel_count: int) -> str:
    if n_components == 1:
        return "single-stroke"
    if n_components <= 3:
        return "well-connected"
    frag_ratio = n_components / max(ink_pixel_count / 1000.0, 1e-6)
    if frag_ratio < 1.0:
        return "mildly-fragmented"
    return "highly-fragmented"


def _kw_stroke_style(binary: np.ndarray) -> str:
    area = int(binary.sum())
    if area == 0:
        return "bold-strokes"
    eroded = binary_erosion(binary)
    perimeter = int((binary & ~eroded).sum())
    pa_ratio = perimeter / area
    if pa_ratio < 0.20:
        return "bold-strokes"
    if pa_ratio < 0.50:
        return "flowing-strokes"
    return "intricate-strokes"


def _kw_pressure_variation(dist: np.ndarray, binary: np.ndarray) -> str:
    ink_dists = dist[binary]
    if ink_dists.size < 2:
        return "uniform-pressure"
    stddev = float(ink_dists.std())
    if stddev < 1.5:
        return "uniform-pressure"
    if stddev < 4.0:
        return "moderate-pressure-variation"
    return "high-pressure-variation"


def _analyse_cv(binary: np.ndarray, canvas_h: int, canvas_w: int) -> list[str]:
    """Run all 11 computer-vision keyword categories and return the results."""
    rows_with_ink = np.where(np.any(binary, axis=1))[0]
    cols_with_ink = np.where(np.any(binary, axis=0))[0]
    bbox_rmin = int(rows_with_ink[0])
    bbox_rmax = int(rows_with_ink[-1])
    bbox_cmin = int(cols_with_ink[0])
    bbox_cmax = int(cols_with_ink[-1])
    bbox_h = bbox_rmax - bbox_rmin + 1
    bbox_w = bbox_cmax - bbox_cmin + 1
    binary_crop = binary[bbox_rmin:bbox_rmax + 1, bbox_cmin:bbox_cmax + 1]
    ink_pixel_count = int(binary.sum())

    dist = distance_transform_edt(binary)
    _, n_components = label(binary)

    return [
        _kw_relative_size(bbox_h, bbox_w, canvas_h, canvas_w),   # A
        _kw_aspect_ratio(bbox_h, bbox_w),                          # B
        _kw_ink_density(binary_crop),                              # C
        _kw_stroke_width(dist, binary),                            # D
        _kw_complexity(n_components),                              # E
        _kw_vertical_com(binary, bbox_rmin, bbox_rmax),            # F
        _kw_height_span(bbox_h, canvas_h),                         # G
        _kw_slant(binary),                                         # H
        _kw_connectivity(n_components, ink_pixel_count),           # I
        _kw_stroke_style(binary),                                  # J
        _kw_pressure_variation(dist, binary),                      # K
    ]

# ── AI inference ──────────────────────────────────────────────────────────────

_AI_PROMPT = (
    "Analyse this handwritten signature image and return ONLY a comma-separated list "
    "of descriptive keywords from everyday language that characterise its visual style, "
    "personality and quality. Choose from attributes such as: neat, messy, rushed, "
    "careful, professional, casual, artistic, plain, bold, timid, confident, hesitant, "
    "elegant, crude, flowing, angular, precise, sloppy, expressive, restrained, "
    "formal, informal, complex, simple, unique, generic, energetic, calm, stylised, "
    "legible, illegible. Return only the keywords, no explanation."
)

_CREATIVE_PROMPT = (
    "Examine this handwritten signature and creatively infer attributes of its author. "
    "Consider: probable gender expression, dominant personality traits, emotional "
    "temperament, confidence level, introversion vs extroversion, creativity vs "
    "analytical mindset, ambition, social tendencies, sensitivity, leadership qualities, "
    "spontaneity vs discipline, and any other characteristics the signature's style, "
    "flow, pressure, rhythm and form suggest about the person behind it. "
    "Be broad and imaginative — this is an interpretive exercise, not a clinical "
    "assessment. Return ONLY a comma-separated list of attribute keywords, no explanation."
)


# Adaptive rate limit buffer
# ----------------------------
# Each rate limit hit adds _RATE_LIMIT_BUFFER_STEP seconds to the wait on top
# of the API-advised retry time and the fixed 1s UX buffer. This accounts for
# the observed pattern where rate limits become more frequent as the token
# window fills up over a long batch. The buffer is capped at _RATE_LIMIT_BUFFER_CAP
# to prevent waits from spiralling, and resets to 0 at the start of each run.
_rate_limit_buffer: float = 0.0
_RATE_LIMIT_BUFFER_STEP: float = 0.5
_RATE_LIMIT_BUFFER_CAP: float = 15.0


def reset_rate_limit_buffer() -> None:
    """Reset the adaptive rate limit buffer. Call at the start of each run."""
    global _rate_limit_buffer
    _rate_limit_buffer = 0.0


def _parse_retry_after(exc: Exception) -> float:
    """Extract the API-suggested wait time in seconds from a rate limit error."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        for header in ("retry-after", "x-ratelimit-reset-requests"):
            value = headers.get(header)
            if value:
                try:
                    return float(value)
                except ValueError:
                    pass
    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(exc), re.IGNORECASE)
    if m:
        value = float(m.group(1))
        return value / 1000 if m.group(2).lower() == "ms" else value
    return 1.0


def _call_vision_api(image_path: pathlib.Path, provider: str, client: Any,
                     prompt: str, *, status_fn: Any = None) -> list[str]:
    """Send image + prompt to the configured vision API; return keyword list.

    Raises RuntimeError on failure so the caller can mark the file as failed.
    """
    if provider == "none" or client is None:
        return []

    img_data = base64.standard_b64encode(image_path.read_bytes()).decode()

    for attempt in range(2):
        model = "claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini"
        logger.info("API call: provider=%s model=%s file=%s attempt=%d",
                    provider, model, image_path.name, attempt + 1)
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=150,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_data,
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                raw = response.content[0].text
            elif provider == "openai":
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=150,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{img_data}",
                        }},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                raw = response.choices[0].message.content
            else:
                return []

            keywords = [kw.strip().lower() for kw in raw.split(",") if kw.strip()]
            logger.info("API response: file=%s keywords=%d", image_path.name, len(keywords))
            return keywords

        except Exception as exc:
            is_rate_limit = (
                "rate" in str(exc).lower()
                or getattr(exc, "status_code", None) == 429
            )
            if attempt == 0 and is_rate_limit:
                global _rate_limit_buffer
                _rate_limit_buffer = min(
                    _rate_limit_buffer + _RATE_LIMIT_BUFFER_STEP,
                    _RATE_LIMIT_BUFFER_CAP,
                )
                wait = _parse_retry_after(exc) + 1.0 + _rate_limit_buffer
                logger.warning("Rate limit hit for %s, waiting %.2fs (buffer=%.1fs): %s",
                               image_path.name, wait, _rate_limit_buffer, exc)
                if status_fn:
                    status_fn(f"Rate limited — waiting {wait:.1f}s (press Stop to cancel)")
                time.sleep(wait)
                continue
            logger.error("API call failed for %s: %s", image_path.name, exc)
            raise RuntimeError(f"API call failed: {exc}") from exc

    raise RuntimeError("API call failed after retry")


def _analyse_ai(image_path: pathlib.Path, provider: str, client: Any,
                *, status_fn: Any = None) -> list[str]:
    """Query vision AI for visual-style keyword descriptors."""
    return _call_vision_api(image_path, provider, client, _AI_PROMPT, status_fn=status_fn)


def _analyse_creative(image_path: pathlib.Path, provider: str, client: Any,
                      *, status_fn: Any = None) -> list[str]:
    """Query vision AI for broad personality and sex-attribute inferences."""
    return _call_vision_api(image_path, provider, client, _CREATIVE_PROMPT, status_fn=status_fn)

# ── Top-level analyser ────────────────────────────────────────────────────────

def analyse_signature(image_path: pathlib.Path, provider: str, client: Any,
                      *, cv: bool = True, style: bool = False,
                      creative: bool = False,
                      status_fn: Any = None) -> list[str]:
    """Return combined keyword list for a signature image.

    CV analysis runs only when cv=True.  AI requests are only made when the
    corresponding flag is True and a client is available.
    status_fn, if provided, is called with a short string before each phase.
    """
    try:
        img = Image.open(image_path).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise RuntimeError(f"Cannot open image: {exc}") from exc

    arr = np.array(img, dtype=np.uint8)
    canvas_h, canvas_w = arr.shape

    binary = arr < 128
    if float(binary.mean()) > 0.5:  # auto-invert dark-background images
        binary = ~binary

    if not binary.any():
        return ["blank-image"]

    keywords: list[str] = []

    if cv:
        if status_fn:
            status_fn("CV Analysis")
        keywords += _analyse_cv(binary, canvas_h, canvas_w)

    if style:
        if status_fn:
            status_fn("Style API query")
        keywords += _analyse_ai(image_path, provider, client, status_fn=status_fn)

    if creative:
        if status_fn:
            status_fn("Creative API query")
        keywords += _analyse_creative(image_path, provider, client, status_fn=status_fn)

    return keywords

# ── Conflict resolution ───────────────────────────────────────────────────────

_GLOBAL_ACTION: str | None = None

_ACTION_MAP = {
    "": "append",
    "a": "append",
    "o": "overwrite",
    "s": "skip",
    "c": "cancel",
}


def resolve_output_conflict(txt_path: pathlib.Path) -> str:
    """Return 'append', 'overwrite', 'skip', or 'cancel'."""
    global _GLOBAL_ACTION

    if not txt_path.exists():
        return "overwrite"

    if _GLOBAL_ACTION is not None:
        return _GLOBAL_ACTION

    # Clear progress bar before prompting
    console.print()

    console.print(f"[warning]![/] [warning]'{txt_path.name}' already exists.[/]")
    console.print("[info][A]ppend  [O]verwrite  [S]kip  [C]ancel  (default: Append)[/]")
    console.print("[info]Add '+' to auto-apply to all remaining files without prompting  (e.g. 'o+')[/]")

    for _ in range(3):
        try:
            raw = console.input("[info]Choice: [/]").strip().lower()
        except EOFError:
            return "skip"

        apply_all = raw.endswith("+")
        key = raw.rstrip("+")

        if key in _ACTION_MAP:
            action = _ACTION_MAP[key]
            if apply_all and action != "cancel":
                _GLOBAL_ACTION = action
                console.print(f"[info]'{action}' will be applied automatically to all remaining conflicts.[/]")
            return action

        console.print(f"[warning]![/] [warning]Unrecognised input '{raw}'. Please enter A, O, S, or C.[/]")

    console.print("[info]No valid input received — defaulting to 'append'.[/]")
    return "append"

# ── File backup ───────────────────────────────────────────────────────────────

def backup_file(txt_path: pathlib.Path) -> pathlib.Path:
    """Copy txt_path into a _backup/ subdirectory; return the backup path."""
    backup_dir = txt_path.parent / "_backup"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / txt_path.name
    shutil.copy2(txt_path, backup_path)
    return backup_path

# ── File writer ───────────────────────────────────────────────────────────────

def write_keywords(txt_path: pathlib.Path, keywords: list[str],
                   action: str) -> tuple[pathlib.Path | None, bool]:
    """Write keywords to file.

    Returns (backup_path, changed):
        backup_path  – path to the backup copy if one was made, else None
        changed      – False when the final keyword list is identical to the
                       existing file content (no write or backup performed)
    """
    if action == "overwrite":
        new_content = ", ".join(keywords) + "\n"
        if txt_path.exists():
            existing_content = txt_path.read_text(encoding="utf-8")
            if new_content == existing_content:
                return None, False
            backup_path = backup_file(txt_path)
        else:
            backup_path = None
        txt_path.write_text(new_content, encoding="utf-8")
        return backup_path, True

    elif action == "append":
        existing_content = txt_path.read_text(encoding="utf-8")
        existing = [
            kw.strip().lower()
            for kw in existing_content.strip().split(",")
            if kw.strip()
        ]
        merged = list(dict.fromkeys(existing + [kw.lower() for kw in keywords]))
        new_content = ", ".join(merged) + "\n"
        if new_content == existing_content:
            return None, False
        backup_path = backup_file(txt_path)
        txt_path.write_text(new_content, encoding="utf-8")
        return backup_path, True

    return None, False

# ── Per-file result printer ───────────────────────────────────────────────────

def print_result(filename: str, keyword_count: int, *, success: bool,
                 skipped: bool = False, no_change: bool = False,
                 reason: str = "") -> None:
    console.print()
    name = f"{filename:<38}"
    count = f"{keyword_count:>3} keywords"
    if skipped:
        console.print(f"  [warning]{name}  {count}  SKIPPED[/]")
    elif no_change:
        console.print(f"  [muted]{name}  {count}  NO CHANGE[/]")
    elif success:
        console.print(f"  [success]✓[/] [success]{name}  {count}  SUCCESS[/]")
    else:
        reason_str = f": {reason}" if reason else ""
        console.print(f"  [error]✗[/] [error]{name}  {count}  FAILED{reason_str}[/]")

# ── Final summary ─────────────────────────────────────────────────────────────

def print_summary(stats: dict[str, Any]) -> None:
    console.print()
    console.rule("[header]Analysis Complete[/]")
    console.print(f"  [info]Total files   [/] : [detail]{stats['total']}[/]")
    console.print(f"  [info]Processed     [/] : [detail]{stats['processed']}[/]")
    if stats["no_change"] > 0:
        console.print(f"  [muted]No change     [/] : [muted]{stats['no_change']}[/]")
    else:
        console.print(f"  [info]No change     [/] : [detail]{stats['no_change']}[/]")
    console.print(f"  [info]Skipped       [/] : [detail]{stats['skipped']}[/]")
    failed = stats["failed"]
    if failed > 0:
        console.print(f"  [error]Failed        [/] : [error]{failed}[/]")
    else:
        console.print(f"  [info]Failed        [/] : [detail]{failed}[/]")
    console.print(f"  [info]Total keywords[/] : [detail]{stats['keywords']}[/]")
    if stats["backed_up"] > 0:
        console.print(f"  [info]Files backed up[/] : [detail]{stats['backed_up']} [info]→[/] [detail]{stats['backup_dir']}[/]")

# ── AI client initialisation ──────────────────────────────────────────────────

def init_ai_client() -> tuple[str, Any]:
    """Return (provider_name, client) based on .env / environment variables."""
    provider_pref = os.environ.get("AI_PROVIDER", "").strip().lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key    = os.environ.get("OPENAI_API_KEY", "").strip()

    _placeholder = "your-key-here"

    def try_anthropic() -> tuple[str, Any] | None:
        if not anthropic_key or anthropic_key == _placeholder:
            return None
        try:
            import anthropic  # noqa: PLC0415
            return ("anthropic", anthropic.Anthropic(api_key=anthropic_key))
        except ImportError:
            console.print("[warning]![/] [warning]'anthropic' package not installed. Run: pip install anthropic[/]")
            return None

    def try_openai() -> tuple[str, Any] | None:
        if not openai_key or openai_key == _placeholder:
            return None
        try:
            import openai  # noqa: PLC0415
            return ("openai", openai.OpenAI(api_key=openai_key))
        except ImportError:
            console.print("[warning]![/] [warning]'openai' package not installed. Run: pip install openai[/]")
            return None

    result: tuple[str, Any] | None
    if provider_pref == "anthropic":
        result = try_anthropic()
    elif provider_pref == "openai":
        result = try_openai()
    else:
        result = try_anthropic() or try_openai()

    if result is None:
        return ("none", None)

    return result

# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sign",
        description="Analyse handwritten signature PNG images and generate descriptive keywords.",
        epilog="Keywords are saved to .txt files sharing the base name of each image.",
    )
    parser.add_argument("input_dir", help="Directory containing PNG signature images")
    parser.add_argument(
        "--style",
        action="store_true",
        default=False,
        help="Use AI to analyse visual style characteristics of the signature "
             "(neat, messy, flowing, etc.) — requires an API key in .env",
    )
    parser.add_argument(
        "--creative",
        action="store_true",
        default=False,
        help="Use AI to infer broad personality and sex attributes from the signature — "
             "requires an API key in .env; can be used independently of or alongside --style",
    )
    args = parser.parse_args()

    p = pathlib.Path(args.input_dir)
    if not p.exists():
        parser.error(f"Path does not exist: {args.input_dir}")
    if not p.is_dir():
        parser.error(f"Not a directory: {args.input_dir}")

    return args

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()
    args = parse_args()

    # ── Title ─────────────────────────────────────────────────────────────────
    console.print()
    console.rule("[header]══  Signature Keyword Analyser  ══[/]")
    console.print()

    # ── AI client init ────────────────────────────────────────────────────────
    if args.style or args.creative:
        provider, client = init_ai_client()
        if provider == "none":
            for flag in ("style", "creative"):
                if getattr(args, flag):
                    console.print(
                        f"[warning]![/] [warning]--{flag} requested but no API key found — "
                        "set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env[/]"
                    )
    else:
        provider, client = "none", None

    # ── Analysis mode status ──────────────────────────────────────────────────
    _W = 14  # width of longest label: "Style analysis"

    def _mode_status(flag_enabled: bool) -> str:
        if not flag_enabled:
            return "[info]disabled[/]"
        if provider != "none":
            return f"[success]enabled[/]  [detail]({provider})[/]"
        return "[info]disabled[/]  [warning](no API key)[/]"

    console.print(f"  [detail]{'CV analysis':<{_W}}[/] : [success]enabled[/]")
    console.print(f"  [detail]{'Style analysis':<{_W}}[/] : {_mode_status(args.style)}")
    console.print(f"  [detail]{'Creative':<{_W}}[/] : {_mode_status(args.creative)}")

    # ── Locate images ─────────────────────────────────────────────────────────
    png_files = find_png_files(args.input_dir)
    if not png_files:
        console.print(f"[error]✗[/] [error]No image files found in '{args.input_dir}'.[/]")
        sys.exit(1)

    total = len(png_files)
    console.print(
        f"\n  [info]Found [/][warning]{total}[/][info] image file(s) in "
        f"[detail]'{args.input_dir}'[/][info].[/]"
    )

    try:
        confirm = console.input(
            "[info]Proceed with analysis? [warning][y/N][/] [info]: [/]"
        ).strip().lower()
    except EOFError:
        confirm = "n"

    if confirm != "y":
        console.print("[info]Aborted.[/]")
        sys.exit(0)

    console.rule("[header]Analysing signatures …[/]")

    stats: dict[str, Any] = {
        "total": total,
        "processed": 0,
        "no_change": 0,
        "skipped": 0,
        "failed": 0,
        "keywords": 0,
        "backed_up": 0,
        "backup_dir": None,
    }

    cancel = False

    try:
        from rich.progress import Progress, BarColumn, TimeRemainingColumn

        with Progress(
            BarColumn(bar_width=40),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[progress]Processing signatures[/]",
                total=total
            )

            for i, img_path in enumerate(png_files):
                if cancel:
                    break

                # Update progress with current filename
                progress.update(task, description=f"[progress]{img_path.name}[/]", advance=1)

                try:
                    keywords = analyse_signature(
                        img_path, provider, client,
                        style=args.style,
                        creative=args.creative,
                    )
                except Exception as exc:
                    print_result(img_path.name, 0, success=False, reason=str(exc))
                    stats["failed"] += 1
                    continue

                txt_path = img_path.with_suffix(".txt")
                action = resolve_output_conflict(txt_path)

                if action == "cancel":
                    cancel = True
                    stats["skipped"] += 1
                    break

                if action == "skip":
                    print_result(img_path.name, len(keywords), success=True, skipped=True)
                    stats["skipped"] += 1
                    continue

                try:
                    backup_path, changed = write_keywords(txt_path, keywords, action)
                except OSError as exc:
                    print_result(img_path.name, 0, success=False, reason=str(exc))
                    stats["failed"] += 1
                    continue

                if not changed:
                    print_result(img_path.name, len(keywords), success=True, no_change=True)
                    stats["no_change"] += 1
                    continue

                if backup_path is not None:
                    stats["backed_up"] += 1
                    stats["backup_dir"] = str(backup_path.parent)

                print_result(img_path.name, len(keywords), success=True)
                stats["processed"] += 1
                stats["keywords"] += len(keywords)

    except KeyboardInterrupt:
        console.print()
        console.print("[warning]![/] [warning]Interrupted by user.[/]")

    print_summary(stats)


if __name__ == "__main__":
    main()
