#!/usr/bin/env python3
"""sign_tui.py — Textual TUI for signature image analysis.

Interactive terminal UI wrapper around sign.py analysis logic.
Uses Textual with the Nord theme for professional appearance.
"""

__version__ = "1.0.0"

import argparse
import logging
import pathlib
import sys
import threading
from typing import Any

from dotenv import load_dotenv
from rich import color
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection
from rich.console import Console
from rich.text import Text

# ── Logging Setup ─────────────────────────────────────────────────────────────

LOG_DIR = pathlib.Path.home() / ".cache" / "sign_tui"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "sign_tui.log"

class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after every record so the log is current."""
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        _FlushingFileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("sign").setLevel(logging.INFO)

logger.info("=" * 70)
logger.info(f"Starting Signature Analyser TUI v{__version__}")
logger.info("=" * 70)


# ── Imports from sign.py ─────────────────────────────────────────────────────

try:
    from sign import (
        analyse_signature,
        find_png_files,
        init_ai_client,
        parse_args,
        reset_rate_limit_buffer,
        write_keywords,
    )

    # Silence sign.py's Rich console output
    import sign as _sign

    _sign.console = Console(quiet=True)
    logger.info("Successfully imported analysis functions from sign.py")
except Exception as e:
    logger.exception("Failed to import from sign.py")
    sys.exit(1)


# ── Messages ──────────────────────────────────────────────────────────────────

class ProgressStatus(Message):
    """Short phase update from the worker for display under the progress bar."""

    def __init__(self, filename: str, phase: str) -> None:
        super().__init__()
        self.filename = filename
        self.phase = phase


class FileStarted(Message):
    """Worker has begun processing a file."""

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename


class FileResult(Message):
    """Worker has finished processing one file."""

    def __init__(
        self,
        filename: str,
        keyword_count: int,
        status: str,
        reason: str = "",
        backup_path: str | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.keyword_count = keyword_count
        self.status = status
        self.reason = reason
        self.backup_path = backup_path
        self.keywords = keywords or []


class WorkerComplete(Message):
    """All files have been processed."""

    pass


class WorkerCancelled(Message):
    """User chose Cancel during conflict resolution."""

    pass


class WorkerAborted(Message):
    """Worker halted due to too many consecutive failures."""

    def __init__(self, reason: str) -> None:
        super().__init__()
        self.reason = reason


class ErrorOccurred(Message):
    """An error occurred during processing."""

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.title = title
        self.message = message


# ── Modals ────────────────────────────────────────────────────────────────────

class ConflictModal(ModalScreen):
    """Conflict resolution modal: Append/Overwrite/Skip/Cancel + Apply to All."""

    BINDINGS = [
        Binding("escape", "cancel_modal", "Cancel", show=False),
        Binding("q", "cancel_modal", "Cancel", show=False),
        Binding("Q", "cancel_modal", "Cancel", show=False),
        Binding("a", "toggle_apply_all", "Apply to all", show=False),
        Binding("A", "toggle_apply_all", "Apply to all", show=False),
    ]

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename

    def action_cancel_modal(self) -> None:
        self.dismiss(("cancel", False))

    def action_toggle_apply_all(self) -> None:
        cb = self.query_one("#chk_apply_all", Checkbox)
        cb.value = not cb.value

    def on_mount(self) -> None:
        self.query_one("#btn_append", Button).focus()

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("File Conflict", classes="modal-header"),
            Static(
                f"[warning]'{self.filename}'[/] already exists.\nChoose how to proceed:",
                classes="modal-message",
            ),
            Checkbox("Apply to all remaining conflicts  (A)", id="chk_apply_all"),
            Horizontal(
                Button("Append", id="btn_append", variant="primary"),
                Button("Overwrite", id="btn_overwrite", variant="warning"),
                Button("Skip", id="btn_skip", variant="default"),
                Button("Cancel", id="btn_cancel", variant="error"),
                classes="button-panel",
            ),
            id="conflict-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        try:
            action_map = {
                "btn_append": "append",
                "btn_overwrite": "overwrite",
                "btn_skip": "skip",
                "btn_cancel": "cancel",
            }
            action = action_map.get(event.button.id, "skip")
            apply_all = self.query_one("#chk_apply_all", Checkbox).value
            self.dismiss((action, apply_all))
        except Exception:
            logger.exception("Error in ConflictModal.on_button_pressed")
            self.dismiss(("cancel", False))


class ErrorModal(ModalScreen):
    """Error display modal."""

    def __init__(self, title: str, message: str) -> None:
        super().__init__()
        self.error_title = title
        self.error_message = message

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"[error]{self.error_title}[/]"),
            Static(f"[info]{self.error_message}[/]"),
            Static("(See ~/.cache/sign_tui/sign_tui.log for details)"),
            Horizontal(
                Button("OK", id="btn_ok", variant="primary"),
                id="error-buttons",
            ),
            id="error-container",
        )

    def on_button_pressed(self, _: Button.Pressed) -> None:
        self.dismiss()


# ── MainScreen ────────────────────────────────────────────────────────────────

class MainScreen(Screen):
    """Main analysis screen with progress, results, and summary."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+s", "stop_processing", "Stop after current", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        # yield Vertical(Static("\nSignature Keyword Analyser\n", id="title"), id="header-container")
        yield Vertical(
                Horizontal(
                    Label(id="method-header"),
                    classes="method-header",
                ),
                SelectionList(
                    Selection("CV analysis", "cv", initial_state=True),
                    Selection("Style analysis", "style", initial_state=False),
                    Selection("Creative", "creative", initial_state=False),
                    id="method-list"
                ),
                # Rule(),
                Container (
                    Button("Start Analysis", id="btn-start", flat=False, classes="wide-buttons"),
                    classes="button-panel",

                ),
                id="method-container",
        )
        # yield Vertical(
            
        #     Static(id="global-action"),
        #     id="info-container",
        # )
        yield Vertical(
            Label("Image Analysis & Keyword Generation — Results", id="results-title"),
            ProgressBar(id="progress"),
            Static(id="current-file"),
            id="progress-container",
        )
        yield DataTable(id="result-table", cursor_type="none")
        yield Vertical(
            Static("Press Ctrl+S to Stop Processing", id="stop-hint"),
            id="stop-hint-container",
        )
        yield Vertical(Static(id="summary-panel"), id="summary-container")
        yield Footer()

    _awaiting_confirm: bool = False
    _stop_requested: bool = False

    def action_stop_processing(self) -> None:
        self._stop_requested = True
        self.query_one("#stop-hint", Static).update("Stopping after current image…")
        logger.info("Graceful stop requested by user")

    def on_mount(self) -> None:
        # self.query_one("#confirm-prompt", Static).styles.display = "none"
        self.query_one("#progress-container").styles.display = "none"
        self.query_one("#result-table", DataTable).styles.display = "none"
        self.query_one("#stop-hint-container").styles.display = "none"
        self.query_one("#summary-container").styles.display = "none"
        # self.query_one("#global-action", Static).styles.display = "none"
        self.query_one("#method-container", None).styles.height = "auto"
        self.query_one("#method-list", SelectionList).border_title = "Select Image Analysis Methods"
        self.app.call_after_refresh(self._init_and_confirm)

    def _init_and_confirm(self) -> None:
        """Load config, find files, show confirmation."""
        try:
            logger.info("Starting initialization")

            # Load dotenv and parse arguments
            load_dotenv()
            self.app.args = parse_args()
            logger.info(f"Parsed args: input_dir={self.app.args.input_dir}, style={self.app.args.style}, creative={self.app.args.creative}")

            # Initialize AI client
            if self.app.args.style or self.app.args.creative:
                try:
                    self.app.provider, self.app.client = init_ai_client()
                    logger.info(f"AI client initialized: provider={self.app.provider}")
                except Exception as e:
                    logger.warning(f"Failed to init AI client: {e}")
                    self.app.provider, self.app.client = "none", None
            else:
                self.app.provider, self.app.client = "none", None
                logger.info("AI disabled via CLI flags")

            # Set initial selection from CLI args
            method_list = self.query_one("#method-list", SelectionList)
            if self.app.args.style:
                method_list.select("style")
            if self.app.args.creative:
                method_list.select("creative")

            # Find PNG files
            try:
                self.app.png_files = find_png_files(str(self.app.args.input_dir))
                logger.info(f"Found {len(self.app.png_files)} PNG files")
            except Exception as e:
                logger.exception(f"Failed to find PNG files: {e}")
                self.app.post_message(ErrorOccurred("File Discovery Error", f"Could not scan directory: {e}"))
                self.app.exit()
                return

            if not self.app.png_files:
                logger.warning("No PNG files found")
                self.app.post_message(ErrorOccurred("No Files", f"No PNG files found in '{self.app.args.input_dir}'"))
                self.app.exit()
                return

            # Update file count
            count = len(self.app.png_files)
            self.query_one("#method-header", Static).update(
                f"Found [b]{count}[/] image(s) in folder: [b]'{self.app.args.input_dir}'[/].\n"
                f"These image file(s) will be analysed based on your selection below..."
                #   methods: " +
                # f"[b]READY TO PROCESS {count}[/][info] image(S) in "
                # f"[detail]'{self.app.args.input_dir}'[/][info].[/]"
            )

            # # Show prompt and await keypress
            # self.query_one("#confirm-prompt", Static).update(
            #     f"Analyse {count} image(s) using CV? [y/n]"
            # )
            # self.query_one("#confirm-prompt", Static).styles.display = "block"
            self._awaiting_confirm = True

        except Exception as e:
            logger.exception("Error during initialization")
            self.app.post_message(ErrorOccurred("Initialization Error", f"Failed to initialize: {e}"))
            self.app.exit()

    def _mode_status(self, flag_enabled: bool) -> str:
        if not flag_enabled:
            return "[muted]disabled[/]"
        if self.app.provider != "none":
            return f"[success]enabled[/] [muted]({self.app.provider})[/]"
        return "[muted]disabled[/] [warning](no API key)[/]"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-start" and self._awaiting_confirm:
            self._awaiting_confirm = False
            self._start_analysis()

    def on_key(self, event) -> None:
        if not self._awaiting_confirm or event.key != "s":
            return
        event.prevent_default()
        self._awaiting_confirm = False
        self._start_analysis()

    def _start_analysis(self) -> None:
        """Read selections, show progress UI, and kick off the worker."""
        try:
            logger.info("User confirmed, starting worker")

            # Read selections and update args
            selected = self.query_one("#method-list", SelectionList).selected

            if not selected:
                self._awaiting_confirm = True
                self.app.notify(
                    "Select at least one analysis method to proceed.",
                    title="No method selected",
                    severity="warning",
                )
                self.query_one("#method-list", SelectionList).focus()
                logger.info("Start blocked: no analysis methods selected")
                return

            self.app.args.cv = "cv" in selected
            self.app.args.style = "style" in selected
            self.app.args.creative = "creative" in selected

            logger.info(
                "Analysis starting: cv=%s style=%s creative=%s",
                self.app.args.cv, self.app.args.style, self.app.args.creative,
            )

            # Re-init AI client if needed and not already done
            if (self.app.args.style or self.app.args.creative) and self.app.provider == "none":
                try:
                    self.app.provider, self.app.client = init_ai_client()
                except Exception:
                    self.app.provider, self.app.client = "none", None

            # Hide the selection list
            self.query_one("#method-container", None).styles.display = "none"

            progress = self.query_one("#progress", ProgressBar)
            progress.update(total=len(self.app.png_files))
            self.query_one("#progress-container").styles.display = "block"
            table = self.query_one("#result-table", DataTable)
            table.add_column("File", key="file", width=25)
            table.add_column("Result", key="result", width=8),
            table.add_column("Count", key="keywords", width=5)
            table.add_column("Keywords", key="words")
            table.styles.display = "block"
            self.query_one("#stop-hint-container").styles.display = "block"
            self.app.stats = {
                "total": len(self.app.png_files),
                "processed": 0,
                "no_change": 0,
                "skipped": 0,
                "failed": 0,
                "keywords": 0,
                "backed_up": 0,
                "backup_dir": None,
            }
            reset_rate_limit_buffer()
            self._run_worker()
        except Exception as e:
            logger.exception("Error starting worker")
            self.post_message(ErrorOccurred("Confirmation Error", f"Failed to start: {e}"))



    def _on_conflict_resolved(self, result: tuple[str, bool] | None) -> None:
        """Callback when ConflictModal is dismissed."""
        try:
            if result is None:
                action, apply_all = "cancel", False
            else:
                action, apply_all = result
            if apply_all and action != "cancel":
                self.app.global_action = action
                logger.info(f"Global action set to: {action}")
            self.app._conflict_result = action
            self.app._conflict_event.set()
        except Exception:
            logger.exception("Error in _on_conflict_resolved")
            self.app._conflict_result = "cancel"
            self.app._conflict_event.set()

    def on_file_started(self, message: FileStarted) -> None:
        """Update UI when a file starts processing."""
        try:
            progress = self.query_one("#progress", ProgressBar)
            progress.update(advance=1)
            modes = []
            if self.app.args.cv:
                modes.append("CV")
            if self.app.args.style:
                modes.append("Style")
            if self.app.args.creative:
                modes.append("Creative")
            mode_str = " · ".join(modes)
            self.query_one("#current-file", Static).update(
                f"Analysing: {message.filename}  ·  {mode_str}"
            )
        except Exception as e:
            logger.exception(f"Error updating UI for file start: {e}")

    def on_progress_status(self, message: ProgressStatus) -> None:
        """Update the progress line with the current analysis phase."""
        try:
            self.query_one("#current-file", Static).update(
                f"{message.filename}  ·  {message.phase}…"
            )
        except Exception:
            logger.exception("Error updating progress status")

    def on_file_result(self, message: FileResult) -> None:
        """Process a file result and update stats."""
        try:
            table = self.query_one("#result-table", DataTable)
            status = message.status
            kw = message.keyword_count

            v = self.app.get_css_variables()
            if status == "success":
                style, result_label = v.get("success", "#4EBF71"), "DONE"
            elif status == "skipped":
                style, result_label = v.get("accent", "#4B9CD3"), "SKIP"
            elif status == "no_change":
                style, result_label = v.get("warning", "#ADFF2F"), "SAME"
            else:
                reason = f": {message.reason[:30]}" if message.reason else ""
                style, result_label = v.get("error", "#FF4444"), "FAIL"

            kw_str = ", ".join(message.keywords) if message.keywords else "—"
            # if len(kw_str) > 67:
            #     kw_str = kw_str[:67] + "…"
            table.add_row(
                Text(message.filename),
                Text(result_label, style=style, justify="center"),
                Text(str(kw) if kw else "—", justify="right"),
                Text(kw_str),
            )

            # Update stats
            if message.status == "success":
                self.app.stats["processed"] += 1
                self.app.stats["keywords"] += message.keyword_count
                if message.backup_path:
                    self.app.stats["backed_up"] += 1
                    self.app.stats["backup_dir"] = str(
                        pathlib.Path(message.backup_path).parent
                    )
            elif message.status == "skipped":
                self.app.stats["skipped"] += 1
            elif message.status == "no_change":
                self.app.stats["no_change"] += 1
            elif message.status == "failed":
                self.app.stats["failed"] += 1

        except Exception as e:
            logger.exception(f"Error processing file result: {e}")

    def _format_summary(self) -> str:
        s = self.app.stats
        no_ch = "dim" if s["no_change"] > 0 else "info"
        fail  = "error" if s["failed"] > 0 else "info"
        return (
            f"[info]Total[/]: [detail]{s['total']}[/]   "
            f"[info]Processed[/]: [detail]{s['processed']}[/]   "
            f"[{no_ch}]No Change[/{no_ch}]: [{no_ch}]{s['no_change']}[/{no_ch}]   "
            f"[info]Skipped[/]: [detail]{s['skipped']}[/]   "
            f"[{fail}]Failed[/{fail}]: [{fail}]{s['failed']}[/{fail}]   "
            f"[info]Keywords Tagged[/]: [detail]{s['keywords']}[/]"
        )

    def on_worker_complete(self, _: WorkerComplete) -> None:
        """Show summary when worker completes."""
        try:
            s = self.app.stats
            self.query_one("#current-file", Static).update(
                f"Analysis complete  ·  {s['total']} image(s) processed  ·  {s['keywords']} keywords tagged"
            )
            self.query_one("#summary-panel", Static).update(self._format_summary())
            self.query_one("#summary-container").styles.display = "block"
            self.query_one("#stop-hint-container").styles.display = "none"
            logger.info("Analysis complete")
        except Exception as e:
            logger.exception("Error displaying completion summary")

    def on_worker_cancelled(self, _: WorkerCancelled) -> None:
        """Exit when user cancels via the conflict modal."""
        logger.info("Analysis cancelled by user")
        self.app.exit()

    def on_worker_aborted(self, message: WorkerAborted) -> None:
        """Show error and exit when consecutive failure ceiling is hit."""
        logger.error("Worker aborted: %s", message.reason)
        self.app.push_screen(
            ErrorModal("Processing Aborted", message.reason),
            lambda _: self.app.exit(),
        )

    def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Display an error modal."""
        try:
            logger.error(f"Error: {message.title} - {message.message}")
            self.app.push_screen(ErrorModal(message.title, message.message))
        except Exception:
            logger.exception("Error displaying error modal")

    def _run_worker(self) -> None:
        """Start the background worker."""
        thread = threading.Thread(target=self._worker_analysis, daemon=True)
        thread.start()
        logger.info("Worker thread started")

    # Consecutive failure ceiling
    # ----------------------------
    # Tracks failures that occur back-to-back during a run. Resets to 0 on any
    # successful file (success, no_change, or skip). Both analysis failures and
    # write failures count toward the streak. Rate limit retries that ultimately
    # succeed do not count. When the ceiling is hit, WorkerAborted is posted,
    # an error modal is shown, and the run halts — a strong signal that something
    # systemic is wrong (e.g. revoked API key, network down).
    _CONSECUTIVE_FAILURE_LIMIT = 5

    def _worker_analysis(self) -> None:
        """Background worker: process all files."""
        app = self.app
        consecutive_failures = 0

        try:
            for idx, img_path in enumerate(app.png_files):
                if self._stop_requested:
                    logger.info("Graceful stop: halting before %s", img_path.name)
                    break
                try:
                    txt_path = img_path.with_suffix(".txt")

                    # Resolve conflict before analysis so we don't make API
                    # calls for files that will be skipped or cancelled.
                    # Global action only applies when a .txt already exists.
                    if not txt_path.exists():
                        action = "overwrite"
                    elif app.global_action is not None:
                        action = app.global_action
                    else:
                        app.call_from_thread(
                            self.post_message,
                            ProgressStatus(img_path.name, "Awaiting Conflict Resolution Response"),
                        )
                        app._conflict_event.clear()
                        app.call_from_thread(
                            app.push_screen,
                            ConflictModal(txt_path.name),
                            self._on_conflict_resolved,
                        )
                        if not app._conflict_event.wait(timeout=60):
                            logger.warning("Conflict resolution timeout")
                            app._conflict_result = "cancel"
                        action = app._conflict_result

                    if action == "cancel":
                        logger.info("User cancelled analysis")
                        app.call_from_thread(self.post_message, WorkerCancelled())
                        return

                    if action == "skip":
                        logger.info("Skipping %s", img_path.name)
                        app.call_from_thread(
                            self.post_message,
                            FileResult(img_path.name, 0, "skipped"),
                        )
                        continue

                    # Post progress update
                    app.call_from_thread(
                        self.post_message,
                        FileStarted(img_path.name),
                    )

                    # Analyse signature
                    def _post_status(phase: str, _name: str = img_path.name) -> None:
                        app.call_from_thread(
                            self.post_message,
                            ProgressStatus(_name, phase),
                        )

                    try:
                        keywords = analyse_signature(
                            img_path,
                            app.provider,
                            app.client,
                            cv=app.args.cv,
                            style=app.args.style,
                            creative=app.args.creative,
                            status_fn=_post_status,
                        )
                    except Exception as exc:
                        logger.exception(f"Analysis failed for {img_path.name}: {exc}")
                        app.call_from_thread(
                            self.post_message,
                            FileResult(img_path.name, 0, "failed", str(exc)),
                        )
                        consecutive_failures += 1
                        if consecutive_failures >= self._CONSECUTIVE_FAILURE_LIMIT:
                            app.call_from_thread(
                                self.post_message,
                                WorkerAborted(
                                    f"Stopped after {self._CONSECUTIVE_FAILURE_LIMIT} consecutive"
                                    " failures — check your API key or logs."
                                ),
                            )
                            return
                        continue

                    # Write keywords
                    try:
                        backup_path, changed = write_keywords(txt_path, keywords, action)
                    except Exception as exc:
                        logger.exception(f"Write failed for {txt_path}: {exc}")
                        app.call_from_thread(
                            self.post_message,
                            FileResult(img_path.name, 0, "failed", str(exc)),
                        )
                        consecutive_failures += 1
                        if consecutive_failures >= self._CONSECUTIVE_FAILURE_LIMIT:
                            app.call_from_thread(
                                self.post_message,
                                WorkerAborted(
                                    f"Stopped after {self._CONSECUTIVE_FAILURE_LIMIT} consecutive"
                                    " failures — check your API key or logs."
                                ),
                            )
                            return
                        continue

                    consecutive_failures = 0
                    status = "no_change" if not changed else "success"
                    app.call_from_thread(
                        self.post_message,
                        FileResult(
                            img_path.name,
                            len(keywords),
                            status,
                            backup_path=str(backup_path) if backup_path else None,
                            keywords=keywords,
                        ),
                    )

                except Exception as e:
                    logger.exception(f"Error processing file {idx}: {e}")
                    app.call_from_thread(
                        self.post_message,
                        FileResult("unknown", 0, "failed", f"Unexpected error: {str(e)[:50]}"),
                    )

            # Signal completion
            app.call_from_thread(
                self.post_message,
                WorkerComplete(),
            )

        except Exception as e:
            logger.exception(f"Worker thread failed: {e}")
            app.call_from_thread(
                self.post_message,
                ErrorOccurred("Worker Error", f"Analysis failed: {str(e)[:100]}"),
            )


# ── SignApp ───────────────────────────────────────────────────────────────────

class SignApp(App):
    """Main Textual application."""

    THEME = "gruvbox"
    TITLE = "Signature Image Analysis and Keyword Allocation"

    CSS_PATH = "image-tag.tcss"

    # State
    args: argparse.Namespace
    provider: str
    client: Any
    png_files: list[pathlib.Path] = []
    global_action: str | None = None
    stats: dict[str, Any] = {}

    # Threading synchronisation
    _conflict_event: threading.Event
    _conflict_result: str = "skip"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def on_mount(self) -> None:
        """Initialize app state."""
        logger.info("SignApp.on_mount() called")
        try:
            self.theme = self.THEME
            self.title = self.TITLE
            self._conflict_event = threading.Event()
            self.push_screen(MainScreen())
            logger.info("Main screen pushed successfully")
        except Exception:
            logger.exception("Failed to mount app")
            self.exit()


if __name__ == "__main__":
    try:
        app = SignApp()
        app.run()
    except Exception as e:
        logger.exception("Fatal error running app")
        print(f"\n[ERROR] Application failed: {e}")
        print(f"[INFO] See ~/.cache/sign_tui/sign_tui.log for details")
        sys.exit(1)
    finally:
        logger.info("Application exiting")
