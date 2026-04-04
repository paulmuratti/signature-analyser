#!/usr/bin/env python3
"""sign_tui.py — Textual TUI for signature image analysis.

Interactive terminal UI wrapper around sign.py analysis logic.
Uses Textual with the Nord theme for professional appearance.
"""

__version__ = "1.0.0"

import argparse
import os
import pathlib
import sys
import threading
import time
from typing import Any

from dotenv import load_dotenv
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    ProgressBar,
    RichLog,
    Static,
)
from rich.console import Console

# Import analysis logic from sign.py
from sign import (
    analyse_signature,
    backup_file,
    find_png_files,
    init_ai_client,
    parse_args,
    write_keywords,
)

# Silence sign.py's Rich console output
import sign as _sign
_sign.console = Console(quiet=True)


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


# ── Modals ────────────────────────────────────────────────────────────────────

class ConfirmScreen(ModalScreen):
    """Confirmation modal: "Proceed with analysis?" Yes/No."""

    def __init__(self, count: int, path: str) -> None:
        super().__init__()
        self.count = count
        self.path = path

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"Found [warning]{self.count}[/] image file(s) in [detail]'{self.path}'[/]"),
            Static("Proceed with analysis?"),
            Horizontal(
                Button("Yes", id="btn_yes", variant="primary"),
                Button("No", id="btn_no", variant="default"),
                id="confirm-buttons",
            ),
            id="confirm-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn_yes")


class ConflictModal(ModalScreen):
    """Conflict resolution modal: Append/Overwrite/Skip/Cancel + Apply to All."""

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static(f"[warning]![/] [warning]'{self.filename}' already exists.[/]"),
            Horizontal(
                Button("Append", id="btn_append", variant="primary"),
                Button("Overwrite", id="btn_overwrite", variant="warning"),
                Button("Skip", id="btn_skip", variant="default"),
                Button("Cancel", id="btn_cancel", variant="error"),
                id="conflict-buttons",
            ),
            Checkbox("Apply to all remaining conflicts", id="chk_apply_all"),
            id="conflict-container",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action_map = {
            "btn_append": "append",
            "btn_overwrite": "overwrite",
            "btn_skip": "skip",
            "btn_cancel": "cancel",
        }
        action = action_map.get(event.button.id, "skip")
        apply_all = self.query_one("#chk_apply_all", Checkbox).value
        self.dismiss((action, apply_all))


# ── MainScreen ────────────────────────────────────────────────────────────────

class MainScreen(Screen):
    """Main analysis screen with progress, results, and summary."""

    CSS = """
    #main-layout {
        height: 1fr;
        padding: 1 2;
    }

    #mode-panel {
        height: auto;
        margin-bottom: 1;
    }

    #mode-panel Static {
        width: 1fr;
    }

    #file-info {
        height: auto;
        margin-bottom: 1;
    }

    #global-action {
        display: none;
        height: 1;
        color: $warning;
        margin-bottom: 1;
    }

    #progress {
        display: none;
        margin-bottom: 1;
    }

    #current-file {
        display: none;
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    #result-log {
        height: 1fr;
        border: solid $surface;
        margin-bottom: 1;
    }

    #summary-panel {
        display: none;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }

    #confirm-container, #conflict-container {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 2 3;
        margin: auto;
    }

    #conflict-buttons {
        height: auto;
        margin-top: 1;
    }

    #conflict-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Horizontal(
                Static("CV analysis", id="label-cv"),
                Static(id="mode-cv"),
                id="mode-cv-row",
            ),
            Horizontal(
                Static("Style analysis", id="label-style"),
                Static(id="mode-style"),
                id="mode-style-row",
            ),
            Horizontal(
                Static("Creative", id="label-creative"),
                Static(id="mode-creative"),
                id="mode-creative-row",
            ),
            id="mode-panel",
        )
        yield Static(id="file-count")
        yield Static(id="global-action")
        yield ProgressBar(total=100, id="progress")
        yield Static(id="current-file")
        yield RichLog(highlight=True, markup=True, id="result-log")
        yield Static(id="summary-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize and show confirmation prompt."""
        self.app.call_after_refresh(self._init_and_confirm)

    def _init_and_confirm(self) -> None:
        """Load config, find files, show confirmation."""
        # Load dotenv and parse arguments
        load_dotenv()
        self.app.args = parse_args()

        # Initialize AI client
        if self.app.args.style or self.app.args.creative:
            self.app.provider, self.app.client = init_ai_client()
        else:
            self.app.provider, self.app.client = "none", None

        # Update mode status
        cv_status = "[success]enabled[/]"
        self.query_one("#mode-cv", Static).update(cv_status)

        style_status = self._mode_status(self.app.args.style)
        self.query_one("#mode-style", Static).update(style_status)

        creative_status = self._mode_status(self.app.args.creative)
        self.query_one("#mode-creative", Static).update(creative_status)

        # Find PNG files
        self.app.png_files = find_png_files(str(self.app.args.input_dir))
        if not self.app.png_files:
            self.app.exit()
            return

        # Update file count
        count = len(self.app.png_files)
        self.query_one("#file-count", Static).update(
            f"\n[info]Found [/][warning]{count}[/][info] image file(s) in "
            f"[detail]'{self.app.args.input_dir}'[/][info].[/]"
        )

        # Show confirmation
        self.app.push_screen(
            ConfirmScreen(count, str(self.app.args.input_dir)),
            self._on_confirmed,
        )

    def _mode_status(self, flag_enabled: bool) -> str:
        """Format mode status line."""
        if not flag_enabled:
            return "[info]disabled[/]"
        if self.app.provider != "none":
            return f"[success]enabled[/]  [detail]({self.app.provider})[/]"
        return "[info]disabled[/]  [warning](no API key)[/]"

    def _on_confirmed(self, confirmed: bool) -> None:
        """User confirmed or cancelled."""
        if not confirmed:
            self.app.exit()
            return

        # Show progress and current file
        self.query_one("#progress", ProgressBar).styles.display = "block"
        self.query_one("#current-file", Static).styles.display = "block"

        # Initialize stats
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

        # Start worker
        self.run_analysis()

    @staticmethod
    def _format_result_line(filename: str, keyword_count: int, status: str, reason: str = "") -> str:
        """Format a per-file result line with Rich markup."""
        name = f"{filename:<38}"
        count = f"{keyword_count:>3} keywords"

        if status == "success":
            return f"  [green]✓[/] [green]{name}[/]  {count}  [green]SUCCESS[/]"
        elif status == "skipped":
            return f"  [yellow]{name}[/]  {count}  [yellow]SKIPPED[/]"
        elif status == "no_change":
            return f"  [dim]{name}[/]  {count}  [dim]NO CHANGE[/]"
        elif status == "failed":
            reason_str = f": {reason}" if reason else ""
            return f"  [red]✗[/] [red]{name}[/]    0 keywords  [red]FAILED{reason_str}[/]"
        return ""

    @staticmethod
    def _format_summary() -> str:
        """Format the final summary panel."""
        app = MainScreen._get_app()
        s = app.stats

        lines = [
            "[header]── Analysis Complete ──[/]",
            "",
            f"  [info]Total files   [/] : [detail]{s['total']}[/]",
            f"  [info]Processed     [/] : [detail]{s['processed']}[/]",
        ]

        if s["no_change"] > 0:
            lines.append(f"  [dim]No change     [/] : [dim]{s['no_change']}[/]")
        else:
            lines.append(f"  [info]No change     [/] : [detail]{s['no_change']}[/]")

        lines.append(f"  [info]Skipped       [/] : [detail]{s['skipped']}[/]")

        if s["failed"] > 0:
            lines.append(f"  [error]Failed        [/] : [error]{s['failed']}[/]")
        else:
            lines.append(f"  [info]Failed        [/] : [detail]{s['failed']}[/]")

        lines.append(f"  [info]Total keywords[/] : [detail]{s['keywords']}[/]")

        if s["backed_up"] > 0:
            lines.append(
                f"  [info]Files backed up[/] : [detail]{s['backed_up']} [info]→[/] [detail]{s['backup_dir']}[/]"
            )

        return "\n".join(lines)

    @staticmethod
    def _get_app() -> "SignApp":
        """Get the app instance."""
        from textual.app import active_app
        return active_app.get()

    def _on_conflict_resolved(self, result: tuple[str, bool]) -> None:
        """Callback when ConflictModal is dismissed."""
        action, apply_all = result
        if apply_all and action != "cancel":
            self.app.global_action = action
            self.query_one("#global-action", Static).update(
                f"[yellow]All conflicts: {action}[/]"
            )
            self.query_one("#global-action").styles.display = "block"
        self.app._conflict_result = action
        self.app._conflict_event.set()

    def on_file_started(self, message: FileStarted) -> None:
        """Update UI when a file starts processing."""
        progress = self.query_one("#progress", ProgressBar)
        progress.update(advance=1)
        self.query_one("#current-file", Static).update(
            f"[muted]Processing: {message.filename}[/]"
        )

    def on_file_result(self, message: FileResult) -> None:
        """Process a file result and update stats."""
        log = self.query_one("#result-log", RichLog)
        line = self._format_result_line(
            message.filename,
            message.keyword_count,
            message.status,
            message.reason,
        )
        log.write(line)

        # Update stats
        if message.status == "success":
            self.app.stats["processed"] += 1
            self.app.stats["keywords"] += message.keyword_count
            if message.backup_path:
                self.app.stats["backed_up"] += 1
                self.app.stats["backup_dir"] = str(pathlib.Path(message.backup_path).parent)
        elif message.status == "skipped":
            self.app.stats["skipped"] += 1
        elif message.status == "no_change":
            self.app.stats["no_change"] += 1
        elif message.status == "failed":
            self.app.stats["failed"] += 1

    def on_worker_complete(self, message: WorkerComplete) -> None:
        """Show summary when worker completes."""
        self.query_one("#current-file", Static).update("")
        self.query_one("#summary-panel", Static).update(self._format_summary())
        self.query_one("#summary-panel").styles.display = "block"

    def on_worker_cancelled(self, message: WorkerCancelled) -> None:
        """Show partial summary when user cancels."""
        self.query_one("#current-file", Static).update("[warning]Cancelled by user[/]")
        self.query_one("#summary-panel", Static).update(self._format_summary())
        self.query_one("#summary-panel").styles.display = "block"

    @staticmethod
    def _action_map(key: str) -> str:
        """Map user input to action."""
        mapping = {
            "": "append",
            "a": "append",
            "o": "overwrite",
            "s": "skip",
            "c": "cancel",
        }
        return mapping.get(key.rstrip("+"), "skip")

    def run_analysis(self) -> None:
        """Start the background worker."""
        self.run_worker(self._worker_analysis)

    def run_worker(self, worker_fn) -> None:
        """Run a worker in the background via the work decorator."""

        def _run():
            worker_fn()

        # Use call_from_thread to safely run the worker in a background thread
        import threading

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _worker_analysis(self) -> None:
        """Background worker: process all files."""
        app = self.app

        for img_path in app.png_files:
            # Post progress update
            app.call_from_thread(
                app.post_message,
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
                app.call_from_thread(
                    app.post_message,
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
                app._conflict_event.wait()
                action = app._conflict_result
            elif app.global_action is not None:
                action = app.global_action
            else:
                action = "overwrite"

            if action == "cancel":
                app.call_from_thread(
                    app.post_message,
                    WorkerCancelled(),
                )
                return

            if action == "skip":
                app.call_from_thread(
                    app.post_message,
                    FileResult(img_path.name, len(keywords), "skipped"),
                )
                continue

            # Write keywords
            try:
                backup_path, changed = write_keywords(txt_path, keywords, action)
            except OSError as exc:
                app.call_from_thread(
                    app.post_message,
                    FileResult(img_path.name, 0, "failed", str(exc)),
                )
                continue

            status = "no_change" if not changed else "success"
            app.call_from_thread(
                app.post_message,
                FileResult(
                    img_path.name,
                    len(keywords),
                    status,
                    backup_path=str(backup_path) if backup_path else None,
                ),
            )

        # Signal completion
        app.call_from_thread(
            app.post_message,
            WorkerComplete(),
        )


# ── SignApp ───────────────────────────────────────────────────────────────────

class SignApp(App):
    """Main Textual application."""

    THEME = "nord"
    TITLE = "Signature Keyword Analyser"

    # State
    args: argparse.Namespace
    provider: str
    client: Any
    png_files: list[pathlib.Path]
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
        self._conflict_event = threading.Event()
        self.push_screen(MainScreen())


if __name__ == "__main__":
    app = SignApp()
    app.run()
