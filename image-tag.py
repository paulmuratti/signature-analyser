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
    Rule,
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

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info(f"Starting Signature Analyser TUI v{__version__}")
logger.info("=" * 70)


# ── Imports from sign_lib.py ──────────────────────────────────────────────────

try:
    from sign_lib import (
        analyse_signature,
        backup_file,
        find_png_files,
        init_ai_client,
        parse_args,
        write_keywords,
    )

    # Silence sign_lib.py's Rich console output
    import sign_lib as _sign_lib

    _sign_lib.console = Console(quiet=True)
    logger.info("Successfully imported analysis functions from sign_lib.py")
except Exception as e:
    logger.exception("Failed to import from sign_lib.py")
    sys.exit(1)


# ── Messages ──────────────────────────────────────────────────────────────────

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
    ) -> None:
        super().__init__()
        self.filename = filename
        self.keyword_count = keyword_count
        self.status = status
        self.reason = reason
        self.backup_path = backup_path


class WorkerComplete(Message):
    """All files have been processed."""

    pass


class WorkerCancelled(Message):
    """User chose Cancel during conflict resolution."""

    pass


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
        except Exception as e:
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

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()


# ── MainScreen ────────────────────────────────────────────────────────────────

class MainScreen(Screen):
    """Main analysis screen with progress, results, and summary."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
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
            Label("Signature Analysis — Results", id="results-title"),
            ProgressBar(id="progress"),
            Static(id="current-file"),
            id="progress-container",
        )
        yield DataTable(id="result-table", cursor_type="none")
        yield Vertical(
            Horizontal(
                Static(id="sum-total",     classes="summary-cell"),
                Static(id="sum-processed", classes="summary-cell"),
                Static(id="sum-no-change", classes="summary-cell"),
            ),
            Horizontal(
                Static(id="sum-skipped",  classes="summary-cell"),
                Static(id="sum-failed",   classes="summary-cell"),
                Static(id="sum-keywords", classes="summary-cell"),
            ),
            id="summary-container",
        )
        yield Footer()

    _awaiting_confirm: bool = False

    def on_mount(self) -> None:
        # self.query_one("#confirm-prompt", Static).styles.display = "none"
        self.query_one("#progress-container").styles.display = "none"
        self.query_one("#result-table", DataTable).styles.display = "none"
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
            self.app.args.style = "style" in selected
            self.app.args.creative = "creative" in selected

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
            table.add_column("File", key="file", width=42)
            table.add_column("Keywords", key="keywords", width=10)
            table.add_column(Text("Result", justify="right"), key="result", width=16)
            table.styles.display = "block"
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
        except Exception as e:
            logger.exception("Error in _on_conflict_resolved")
            self.app._conflict_result = "cancel"
            self.app._conflict_event.set()

    def on_file_started(self, message: FileStarted) -> None:
        """Update UI when a file starts processing."""
        try:
            progress = self.query_one("#progress", ProgressBar)
            progress.update(advance=1)
            modes = ["CV"]
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

    def on_file_result(self, message: FileResult) -> None:
        """Process a file result and update stats."""
        try:
            table = self.query_one("#result-table", DataTable)
            status = message.status
            kw = message.keyword_count

            if status == "success":
                style, result_label = "green", "✓  SUCCESS"
            elif status == "skipped":
                style, result_label = "yellow", "SKIPPED"
            elif status == "no_change":
                style, result_label = "dim", "NO CHANGE"
            else:
                reason = f": {message.reason[:30]}" if message.reason else ""
                style, result_label = "red", f"✗  FAILED{reason}"

            table.add_row(
                Text(message.filename, style=style),
                Text(str(kw) if kw else "—", style=style, justify="right"),
                Text(result_label, style=style, justify="right"),
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

    def on_worker_complete(self, message: WorkerComplete) -> None:
        """Show summary when worker completes."""
        try:
            s = self.app.stats
            self.query_one("#current-file", Static).update(
                f"Analysis complete  ·  {s['total']} image(s) processed  ·  {s['keywords']} keywords tagged"
            )
            no_ch = "dim" if s["no_change"] > 0 else "info"
            fail  = "error" if s["failed"] > 0 else "info"
            self.query_one("#sum-total",     Static).update(f"[info]Total Files[/]\n[detail]{s['total']}[/]")
            self.query_one("#sum-processed", Static).update(f"[info]Processed[/]\n[detail]{s['processed']}[/]")
            self.query_one("#sum-no-change", Static).update(f"[{no_ch}]No Change[/{no_ch}]\n[{no_ch}]{s['no_change']}[/{no_ch}]")
            self.query_one("#sum-skipped",   Static).update(f"[info]Skipped[/]\n[detail]{s['skipped']}[/]")
            self.query_one("#sum-failed",    Static).update(f"[{fail}]Failed[/{fail}]\n[{fail}]{s['failed']}[/{fail}]")
            self.query_one("#sum-keywords",  Static).update(f"[info]Keywords Tagged[/]\n[detail]{s['keywords']}[/]")
            self.query_one("#summary-container").styles.display = "block"
            logger.info("Analysis complete")
        except Exception as e:
            logger.exception("Error displaying completion summary")

    def on_worker_cancelled(self, message: WorkerCancelled) -> None:
        """Exit when user cancels via the conflict modal."""
        logger.info("Analysis cancelled by user")
        self.app.exit()

    def on_error_occurred(self, message: ErrorOccurred) -> None:
        """Display an error modal."""
        try:
            logger.error(f"Error: {message.title} - {message.message}")
            self.app.push_screen(ErrorModal(message.title, message.message))
        except Exception as e:
            logger.exception("Error displaying error modal")

    def _run_worker(self) -> None:
        """Start the background worker."""
        app = self.app
        thread = threading.Thread(target=self._worker_analysis, daemon=True)
        thread.start()
        logger.info("Worker thread started")

    def _worker_analysis(self) -> None:
        """Background worker: process all files."""
        app = self.app

        try:
            for idx, img_path in enumerate(app.png_files):
                try:
                    # Post progress update
                    app.call_from_thread(
                        self.post_message,
                        FileStarted(img_path.name),
                    )

                    # Analyse signature
                    try:
                        keywords = analyse_signature(
                            img_path,
                            app.provider,
                            app.client,
                            style=app.args.style,
                            creative=app.args.creative,
                        )
                    except Exception as exc:
                        logger.exception(f"Analysis failed for {img_path.name}: {exc}")
                        app.call_from_thread(
                            self.post_message,
                            FileResult(img_path.name, 0, "failed", str(exc)),
                        )
                        continue

                    # Resolve conflict
                    txt_path = img_path.with_suffix(".txt")
                    if txt_path.exists() and app.global_action is None:
                        # Clear event, push modal, wait
                        app._conflict_event.clear()
                        app.call_from_thread(
                            app.push_screen,
                            ConflictModal(txt_path.name),
                            self._on_conflict_resolved,
                        )
                        # Wait for modal resolution (with timeout to prevent hanging)
                        if not app._conflict_event.wait(timeout=60):
                            logger.warning("Conflict resolution timeout")
                            app._conflict_result = "cancel"
                        action = app._conflict_result
                    elif app.global_action is not None:
                        action = app.global_action
                    else:
                        action = "overwrite"

                    if action == "cancel":
                        logger.info("User cancelled analysis")
                        app.call_from_thread(
                            self.post_message,
                            WorkerCancelled(),
                        )
                        return

                    if action == "skip":
                        app.call_from_thread(
                            self.post_message,
                            FileResult(img_path.name, len(keywords), "skipped"),
                        )
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
                        continue

                    status = "no_change" if not changed else "success"
                    app.call_from_thread(
                        self.post_message,
                        FileResult(
                            img_path.name,
                            len(keywords),
                            status,
                            backup_path=str(backup_path) if backup_path else None,
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
        except Exception as e:
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
