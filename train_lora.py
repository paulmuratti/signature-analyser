#!/usr/bin/env python3
"""train_lora.py — SDXL LoRA training launcher for signature images.

Generates a kohya_ss dataset TOML config, then launches
sdxl_train_network.py via accelerate.  Attention is routed through
PyTorch SDPA so that no xformers kernel is required.

Usage:
    python train_lora.py                   # train with defaults
    python train_lora.py --dry-run         # print command, do not execute
    python train_lora.py --dataset ./dataset --output ./lora_output
    python train_lora.py --epochs 20 --rank 64 --batch-size 1
"""

__version__ = "1.0.0"

import argparse
import os
import pathlib
import subprocess
import sys
import textwrap

# ── Colorama shim ─────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style
    from colorama import init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:
    class _Noop:
        def __getattr__(self, _: str) -> str:
            return ""
    Fore = Style = _Noop()  # type: ignore[assignment]

C_OK     = Fore.GREEN
C_FAIL   = Fore.RED
C_WARN   = Fore.YELLOW
C_INFO   = Fore.CYAN
C_HEAD   = Fore.MAGENTA
C_RESET  = Style.RESET_ALL

def _ok(msg: str)     -> None: print(f"{C_OK}  \u2713  {msg}{C_RESET}")
def _info(msg: str)   -> None: print(f"{C_INFO}{msg}{C_RESET}")
def _warn(msg: str)   -> None: print(f"{C_WARN}  !  {msg}{C_RESET}")
def _err(msg: str)    -> None: print(f"{C_FAIL}  \u2717  {msg}{C_RESET}", file=sys.stderr)
def _header(msg: str) -> None:
    bar = "\u2550" * 60
    print(f"\n{C_HEAD}{bar}\n  {msg}\n{bar}{C_RESET}")

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION  ─  edit this block to tune the training run
# ═══════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────
KOHYA_DIR        = pathlib.Path.home() / "Dev" / "kohya_ss"
SCRIPTS_DIR      = KOHYA_DIR / "sd-scripts"
VENV_ACCELERATE  = KOHYA_DIR / ".venv" / "bin" / "accelerate"
TRAIN_SCRIPT     = SCRIPTS_DIR / "sdxl_train_network.py"

HERE             = pathlib.Path(__file__).parent.resolve()
DEFAULT_DATASET  = HERE / "dataset"
DEFAULT_OUTPUT   = HERE / "lora_output"
CONFIG_DIR       = HERE / "lora_config"
LOGGING_DIR      = HERE / "lora_logs"

# ── Base model ────────────────────────────────────────────────────
BASE_MODEL       = "stabilityai/stable-diffusion-xl-base-1.0"

# ── LoRA network ──────────────────────────────────────────────────
NETWORK_MODULE   = "networks.lora"
NETWORK_DIM      = 32     # rank — higher = more capacity & VRAM
NETWORK_ALPHA    = 16     # scaling (dim/2 is a common starting point)

# ── Resolution & bucketing ────────────────────────────────────────
# Sample images are 725×359 (≈2:1 aspect).  Bucketing lets kohya_ss
# resize each image to the nearest valid resolution bucket so that
# no detail is lost to forced square cropping.
RESOLUTION         = 1024   # SDXL native training resolution
ENABLE_BUCKET      = True
MIN_BUCKET_RESO    = 256
MAX_BUCKET_RESO    = 1024
BUCKET_RESO_STEPS  = 32

# ── Dataset ───────────────────────────────────────────────────────
CAPTION_EXTENSION  = ".txt"
NUM_REPEATS        = 10     # repetitions of each image per epoch

# ── Training schedule ─────────────────────────────────────────────
# 82 images × 10 repeats × 15 epochs / batch 2 ≈ 6 150 steps
BATCH_SIZE         = 2
MAX_EPOCHS         = 15
SAVE_EVERY_EPOCHS  = 5
GRAD_ACCUM_STEPS   = 1     # increase to simulate larger batch sizes
SEED               = 42

# ── Learning rates ────────────────────────────────────────────────
UNET_LR             = 1e-4
TEXT_ENCODER_LR     = 5e-5
LR_SCHEDULER        = "cosine_with_restarts"
LR_SCHEDULER_CYCLES = 3    # restarts for cosine schedule
LR_WARMUP_RATIO     = 0.05 # fraction of total steps used for warm-up

# ── Precision & optimiser ─────────────────────────────────────────
# AdamW8bit requires bitsandbytes (present in kohya_ss venv).
# Fall back to "AdamW" if you encounter import errors.
OPTIMIZER          = "AdamW8bit"
MIXED_PRECISION    = "bf16"  # bf16 is stable on Ampere+; use fp16 on Turing

# ── Output naming ─────────────────────────────────────────────────
OUTPUT_NAME        = "signature_xl_lora"

# ═══════════════════════════════════════════════════════════════════
#  END OF CONFIGURATION
# ═══════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch SDXL LoRA training for signature images via kohya_ss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    type=pathlib.Path, default=DEFAULT_DATASET,
                   help="Directory containing PNG images + .txt caption files")
    p.add_argument("--output",     type=pathlib.Path, default=DEFAULT_OUTPUT,
                   help="Directory to write the trained LoRA safetensors")
    p.add_argument("--epochs",     type=int,   default=MAX_EPOCHS,
                   help="Number of training epochs")
    p.add_argument("--rank",       type=int,   default=NETWORK_DIM,
                   help="LoRA rank (network_dim)")
    p.add_argument("--alpha",      type=float, default=NETWORK_ALPHA,
                   help="LoRA alpha (network_alpha)")
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE,
                   help="Training batch size per GPU")
    p.add_argument("--repeats",    type=int,   default=NUM_REPEATS,
                   help="Dataset repetitions per epoch")
    p.add_argument("--name",       type=str,   default=OUTPUT_NAME,
                   help="Base name for saved LoRA files")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print the training command without executing it")
    return p.parse_args()


def preflight(args: argparse.Namespace) -> bool:
    """Verify that all required paths and files are present."""
    _header("Pre-flight checks")
    ok = True

    checks = [
        (KOHYA_DIR,         "kohya_ss directory"),
        (SCRIPTS_DIR,       "kohya_ss sd-scripts directory"),
        (VENV_ACCELERATE,   "accelerate binary (.venv)"),
        (TRAIN_SCRIPT,      "sdxl_train_network.py"),
        (args.dataset,      "dataset directory"),
    ]
    for path, label in checks:
        if path.exists():
            _ok(f"{label}: {path}")
        else:
            _err(f"{label} not found: {path}")
            ok = False

    if ok:
        pngs = list(args.dataset.glob("*.png")) + list(args.dataset.glob("*.PNG"))
        if pngs:
            _ok(f"Found {len(pngs)} PNG file(s) in dataset")
        else:
            _err(f"No PNG files found in {args.dataset}")
            ok = False

    return ok


def generate_dataset_toml(args: argparse.Namespace) -> pathlib.Path:
    """Write a kohya_ss dataset TOML config and return its path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    toml_path = CONFIG_DIR / "dataset.toml"

    content = textwrap.dedent(f"""\
        # kohya_ss dataset config — generated by train_lora.py
        # Docs: https://github.com/kohya-ss/sd-scripts/blob/main/docs/config_README-ja.md

        [[datasets]]
        resolution           = {RESOLUTION}
        batch_size           = {args.batch_size}
        enable_bucket        = {str(ENABLE_BUCKET).lower()}
        min_bucket_reso      = {MIN_BUCKET_RESO}
        max_bucket_reso      = {MAX_BUCKET_RESO}
        bucket_reso_steps    = {BUCKET_RESO_STEPS}
        bucket_no_upscale    = false   # allow upscaling to fill buckets

          [[datasets.subsets]]
          image_dir          = "{args.dataset.resolve()}"
          caption_extension  = "{CAPTION_EXTENSION}"
          num_repeats        = {args.repeats}
    """)

    toml_path.write_text(content, encoding="utf-8")
    _ok(f"Dataset config written: {toml_path}")
    return toml_path


def build_command(args: argparse.Namespace, toml_path: pathlib.Path) -> list[str]:
    """Construct the full accelerate launch command."""
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    # Total training steps (approximate, used for warmup ratio)
    pngs = list(args.dataset.glob("*.png")) + list(args.dataset.glob("*.PNG"))
    total_steps = max(1, (len(pngs) * args.repeats * args.epochs) // args.batch_size)
    warmup_steps = max(1, int(total_steps * LR_WARMUP_RATIO))

    accel_args = [
        str(VENV_ACCELERATE), "launch",
        f"--num_cpu_threads_per_process=2",
        f"--mixed_precision={MIXED_PRECISION}",
    ]

    train_args = [
        str(TRAIN_SCRIPT),

        # ── Model ─────────────────────────────────────────────────
        f"--pretrained_model_name_or_path={BASE_MODEL}",

        # ── Dataset ───────────────────────────────────────────────
        f"--dataset_config={toml_path}",

        # ── Output ────────────────────────────────────────────────
        f"--output_dir={output_dir}",
        f"--output_name={args.name}",
        "--save_model_as=safetensors",
        f"--save_every_n_epochs={SAVE_EVERY_EPOCHS}",
        f"--save_precision={MIXED_PRECISION}",

        # ── LoRA network ──────────────────────────────────────────
        f"--network_module={NETWORK_MODULE}",
        f"--network_dim={args.rank}",
        f"--network_alpha={args.alpha}",

        # ── Learning rates ────────────────────────────────────────
        f"--unet_lr={UNET_LR}",
        f"--text_encoder_lr={TEXT_ENCODER_LR}",
        f"--learning_rate={UNET_LR}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_scheduler_num_cycles={LR_SCHEDULER_CYCLES}",
        f"--lr_warmup_steps={warmup_steps}",

        # ── Training schedule ─────────────────────────────────────
        f"--max_train_epochs={args.epochs}",
        f"--train_batch_size={args.batch_size}",
        f"--gradient_accumulation_steps={GRAD_ACCUM_STEPS}",
        f"--seed={SEED}",

        # ── Precision ─────────────────────────────────────────────
        f"--mixed_precision={MIXED_PRECISION}",
        "--no_half_vae",               # keep VAE in fp32 to avoid NaN with bf16

        # ── Attention — PyTorch SDPA, no xformers ─────────────────
        "--sdpa",

        # ── Memory efficiency ─────────────────────────────────────
        f"--optimizer_type={OPTIMIZER}",
        "--gradient_checkpointing",    # trade compute for VRAM
        "--cache_latents",             # pre-encode images with the VAE
        "--cache_latents_to_disk",     # persist latent cache across restarts

        # ── Logging ───────────────────────────────────────────────
        f"--logging_dir={LOGGING_DIR}",
        "--log_with=tensorboard",
        f"--log_prefix={args.name}_",

        # ── Misc ──────────────────────────────────────────────────
        "--max_data_loader_n_workers=2",
        "--persistent_data_loader_workers",
    ]

    return accel_args + train_args


def print_summary(args: argparse.Namespace, cmd: list[str]) -> None:
    """Print a human-readable summary of the training run parameters."""
    _header("Training configuration")

    pngs = list(args.dataset.glob("*.png")) + list(args.dataset.glob("*.PNG"))
    n_images = len(pngs)
    total_steps = max(1, (n_images * args.repeats * args.epochs) // args.batch_size)

    rows = [
        ("Base model",        BASE_MODEL),
        ("Dataset",           f"{args.dataset}  ({n_images} PNG files)"),
        ("Output",            str(args.output)),
        ("LoRA name",         args.name),
        ("Rank / alpha",      f"{args.rank} / {args.alpha}"),
        ("Resolution",        f"{RESOLUTION}px  (bucketed, min {MIN_BUCKET_RESO} – max {MAX_BUCKET_RESO})"),
        ("Batch size",        str(args.batch_size)),
        ("Repeats / epoch",   str(args.repeats)),
        ("Epochs",            str(args.epochs)),
        ("Est. total steps",  str(total_steps)),
        ("Optimizer",         OPTIMIZER),
        ("LR  U-Net / TE",    f"{UNET_LR}  /  {TEXT_ENCODER_LR}"),
        ("LR scheduler",      f"{LR_SCHEDULER}  (cycles={LR_SCHEDULER_CYCLES})"),
        ("Mixed precision",   MIXED_PRECISION),
        ("Attention",         "PyTorch SDPA  (--sdpa)"),
        ("Save every",        f"{SAVE_EVERY_EPOCHS} epochs"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, val in rows:
        print(f"  {C_INFO}{key:<{width}}{C_RESET}  {val}")

    _header("Accelerate launch command")
    # Pretty-print the command with line continuations
    indent = "    "
    parts = [cmd[0]]
    for tok in cmd[1:]:
        if tok.startswith("--") or tok == str(TRAIN_SCRIPT):
            parts.append(f"\\\n{indent}{tok}")
        else:
            parts[-1] += f" {tok}"
    print(f"{C_INFO}" + " ".join(parts[:3]) + C_RESET)
    print(f"{C_INFO}" + "\\\n".join(parts[3:]) + C_RESET)


def main() -> None:
    args = parse_args()

    if not preflight(args):
        sys.exit(1)

    toml_path = generate_dataset_toml(args)
    cmd = build_command(args, toml_path)
    print_summary(args, cmd)

    if args.dry_run:
        print(f"\n{C_WARN}  Dry run — command printed above, not executed.{C_RESET}\n")
        return

    _header("Launching training")
    _info("  Training output will stream below. "
          "Press Ctrl-C to interrupt.\n"
          "  Monitor with:  tensorboard --logdir lora_logs\n")

    # Run from sd-scripts/ so that 'networks.lora' and 'library.*' imports resolve.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS_DIR)

    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR), env=env)

    print()
    if result.returncode == 0:
        _header("Training complete")
        _ok(f"LoRA saved to:  {args.output}")
        _info(f"  Files: {args.name}_epoch-*.safetensors")
        _info(f"  Logs:  {LOGGING_DIR}  (tensorboard --logdir lora_logs)")
    else:
        _header("Training failed")
        _err(f"Process exited with code {result.returncode}")
        _warn("Common causes:")
        _warn("  • Out of VRAM — try --batch-size 1 or --rank 16")
        _warn("  • Missing dependency — run: pip install -r requirements.txt")
        _warn("  • xformers conflict — ensure --sdpa is passed (it is by default)")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
