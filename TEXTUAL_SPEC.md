# Textual Standard Specification

## Overview
Standardized terminal UI framework for building professional, feature-rich CLI applications with comprehensive theming and pre-styled components. Textual handles colors, typography, layout, and spacing cohesively — no custom spec building needed.

---

## Library Selection

| Use Case | Library | Rationale |
|----------|---------|-----------|
| Simple CLI with styled output, progress bars | **Rich** | Lightweight, output-only (use TUI_SPEC.md) |
| Interactive TUI, dashboards, forms, complex layouts | **Textual** | Professional pre-built themes, CSS styling, components |

This spec covers **Textual**. For simple output-only CLIs, use Rich (TUI_SPEC.md).

---

## Textual Themes

Textual ships with professional, battle-tested themes. Pick one — no customization needed:

| Theme | Style | Best For |
|-------|-------|----------|
| **Nord** | Cool, muted pastels (Artic & Snow palette) | Modern, calm aesthetics |
| **Dracula** | High contrast, dark purples (Vampire theme) | High visibility, dramatic |
| **Monokai** | Dark with bright accents (code editor style) | Developer-focused tools |
| **Solarized Dark** | Low-contrast, warm tones | Reduced eye strain |
| **Solarized Light** | Light variant of Solarized | Bright environments |
| **Vim** | Neutral, vim-inspired | Minimal, distraction-free |

All themes are **complete** — they define colors, borders, spacing, typography alignment automatically.

---

## Basic App Structure

```python
#!/usr/bin/env python3
from textual.app import ComposeResult, Screen
from textual.containers import Container, Vertical
from textual.widgets import Header, Footer, Static, Button, Input
from textual.binding import Binding

class MyApp(Screen):
    """Main application screen."""
    
    TITLE = "My Application"
    CSS_PATH = "style.tcss"  # Optional: custom styling
    THEME = "nord"  # Pick: nord, dracula, monokai, vim, solarized-dark, solarized-light
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Container(
            Static("Welcome", id="title"),
            Input(placeholder="Enter something"),
            Button("Process", id="btn_process"),
            id="main",
        )
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "btn_process":
            self.query_one(Static).update("Processing...")

if __name__ == "__main__":
    app = MyApp()
    app.run()
```

Set `THEME` to one of the built-in theme names — that's it. Everything else inherits the theme.

---

## Pre-Built Components

Textual provides rich, styled components. All use the theme automatically:

### Text Display
```python
from textual.widgets import Static, Label, RichLog

# Static text (block)
Static("Content here", id="content")

# Label (inline)
Label("Status: [bold green]Ready[/]")

# Rich-formatted log
log = RichLog(highlight=True, markup=True)
log.write("[success]✓ Operation complete[/]")
```

### Input
```python
from textual.widgets import Input, TextArea

# Single-line input
Input(placeholder="Enter name", id="name_input")

# Multi-line text editor
TextArea(language="python", id="code_editor")
```

### Selection
```python
from textual.widgets import Select, Checkbox

# Dropdown
Select(
    [("Option 1", 1), ("Option 2", 2)],
    id="dropdown",
    value=1,
)

# Checkbox
Checkbox("Enable feature", id="feature_toggle")
```

### Data Display
```python
from textual.widgets import DataTable

table = DataTable()
table.add_column("Name", key="name")
table.add_column("Status", key="status")
table.add_row("Item 1", "Ready", key="row1")
```

### Layout
```python
from textual.containers import Container, Horizontal, Vertical, Tabs

# Vertical stack (default)
Vertical(widget1, widget2, widget3)

# Horizontal layout
Horizontal(widget1, widget2, widget3)

# Tabbed interface
Tabs(
    ("Tab 1", widget1),
    ("Tab 2", widget2),
)
```

### Progress
```python
from textual.widgets import ProgressBar

progress = ProgressBar(total=100, id="progress")
progress.update(advance=10)
```

---

## Styling System (CSS-like)

Create `style.tcss` (Textual CSS) for custom styling beyond themes:

```css
/* Containers */
#main {
    width: 100%;
    height: 100%;
    border: solid $primary;
}

/* Text styling */
#title {
    color: $text;
    text-align: center;
    width: 100%;
    height: 3;
    border: solid $accent;
}

/* Input focus */
Input:focus {
    border: heavy $success;
    color: $success;
}

/* Buttons */
Button {
    margin: 1 2;
}

Button:hover {
    background: $accent;
}
```

Reference it in your app:
```python
class MyApp(Screen):
    CSS_PATH = "style.tcss"
    THEME = "nord"
```

**Note:** Most projects don't need custom CSS — the theme + built-in components are sufficient.

---

## Built-in Theme Colors (via CSS variables)

All themes define these color variables automatically. Use in custom CSS if needed:

```css
/* Available in all themes */
color: $primary;      /* Main brand color */
color: $secondary;    /* Secondary color */
color: $accent;       /* Accent highlight */
color: $text;         /* Standard text */
color: $text-muted;   /* De-emphasized text */
color: $success;      /* Success green */
color: $warning;      /* Warning yellow */
color: $error;        /* Error red */
color: $surface;      /* Background */
```

Example:
```python
from textual.widgets import Static
from textual.reactive import reactive

status = Static(renderable="[bold]Ready[/]")
status.styles.color = "green"  # or "$success" in CSS
```

---

## Common Patterns

### Modal Dialog
```python
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Button, Static
from textual.screen import Screen

class ConfirmDialog(Screen):
    def compose(self) -> ComposeResult:
        yield Static("Are you sure?", id="prompt")
        yield Container(
            Button("Yes", id="btn_yes", variant="primary"),
            Button("No", id="btn_no", variant="default"),
            id="buttons",
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        result = event.button.id == "btn_yes"
        self.app.pop_screen(result)
```

Use it:
```python
def action_confirm(self) -> None:
    def handle_result(confirmed: bool) -> None:
        if confirmed:
            self.process()
    
    self.app.push_screen(ConfirmDialog(), handle_result)
```

### Status Bar
```python
from textual.containers import Horizontal
from textual.widgets import Static

class StatusBar(Static):
    def render(self) -> str:
        return f"[bold]Status:[/] Ready | [dim]{self.app.title}[/]"
```

### Progress Indicator
```python
from textual.widgets import ProgressBar
from textual.containers import Vertical

Vertical(
    Static("Processing files..."),
    ProgressBar(total=100, id="progress"),
)
```

---

## Example: Full Application

```python
#!/usr/bin/env python3
from textual.app import ComposeResult, Screen
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Static, Input, Button, DataTable
from textual.binding import Binding

class AnalysisScreen(Screen):
    TITLE = "Signature Analysis"
    THEME = "nord"  # Professional, calm theme
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Container(
            Static("[bold]Signature Keyword Analyzer[/]", id="title"),
            Vertical(
                Static("Select directory:", id="label"),
                Input(placeholder="./dataset", id="dir_input"),
                Horizontal(
                    Button("Analyze", id="btn_analyze", variant="primary"),
                    Button("Clear", id="btn_clear"),
                ),
                id="input_section",
            ),
            Vertical(
                Static("Results", id="results_header"),
                DataTable(id="results_table"),
                id="results_section",
            ),
            id="main",
        )
        yield Footer()
    
    def on_mount(self) -> None:
        """Set up the results table."""
        table = self.query_one(DataTable)
        table.add_columns("File", "Keywords", "Status")
        table.cursor_type = "row"
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button actions."""
        if event.button.id == "btn_analyze":
            self.action_analyze()
        elif event.button.id == "btn_clear":
            self.action_clear()
    
    def action_analyze(self) -> None:
        """Start analysis."""
        dir_input = self.query_one(Input)
        self.query_one(Static, "#results_header").update(
            "[bold green]✓ Analyzing...[/]"
        )
        # Implement analysis logic here
    
    def action_clear(self) -> None:
        """Clear results."""
        self.query_one(DataTable).clear()
    
    def action_refresh(self) -> None:
        """Refresh display."""
        pass

if __name__ == "__main__":
    app = AnalysisScreen()
    app.run()
```

Run:
```bash
python3 app.py
```

The Nord theme automatically applies to all widgets — no color tweaking needed.

---

## Installation & Requirements

```bash
pip install textual>=0.50.0
```

Add to `requirements.txt`:
```
textual>=0.50.0
```

---

## Deployment Checklist

When requesting a Textual app:

- [ ] Specify theme upfront: `THEME = "nord"` (or dracula/monokai/vim/solarized-*)
- [ ] Use pre-built components (don't style from scratch)
- [ ] Custom CSS only if theme + defaults don't fit your needs
- [ ] Test in target terminal (colors vary by terminal emulator)
- [ ] Bindings for common actions (quit, refresh, help)

Example request:
```
Build a Textual app that:
- Uses the "dracula" theme
- Shows a file browser, analysis results table, progress bar
- Has keyboard bindings for common actions
- No custom CSS needed unless defaults don't work
```

---

## When to Use Textual vs Rich

| Scenario | Use |
|----------|-----|
| Simple CLI, one-way output, progress bar | Rich (TUI_SPEC.md) |
| Interactive menus, multi-screen app, forms | **Textual** |
| Real-time dashboard, data monitoring | **Textual** |
| Wizard/survey flow | Rich (TUI_SPEC.md) or Questionary |

---

## Resources

- **Docs:** https://textual.textualize.io/
- **Theme Gallery:** https://textual.textualize.io/guide/themes/
- **Component Guide:** https://textual.textualize.io/widgets/
- **CSS Reference:** https://textual.textualize.io/guide/styles/

Textual is actively maintained, well-documented, and battle-tested in production applications.
