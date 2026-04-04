# TUI Standard Specification

## Overview
Standardized terminal UI library, theme, and output patterns to reduce iteration cycles on console styling.

---

## Library Selection

| Use Case | Library | Rationale |
|----------|---------|-----------|
| CLI with styled output, progress bars, tables | **Rich** | Low boilerplate, proven themes, deterministic output |
| Interactive multi-screen TUI | **Textual** | Full component framework (use only if explicitly approved) |

This spec covers **Rich**. Textual specs to follow if needed.

---

## Rich Theme

```python
from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "info": "white",                  # standard messages, labels
    "detail": "bright_white",         # values, paths, emphasis
    "success": "bright_green",        # success states, checkmarks
    "warning": "bright_yellow",       # warnings, cautions
    "error": "bright_red",            # errors, failures
    "header": "bright_magenta",       # section headers, titles
    "progress": "bright_blue",        # progress indicators
    "muted": "dim white",             # secondary info, de-emphasized
})

console = Console(theme=THEME)
```

**Install:** Add `rich>=13.0.0` to `requirements.txt`

---

## Output Patterns

### 1. Headers / Section Breaks
```python
console.rule("[header]Section Title[/]")
```
Renders as:
```
════════════════════════════════════════ Section Title ════════════════════════════════════════
```

### 2. Status Lines (multi-color)
Use `console.print()` with markup:
```python
console.print(
    f"[info]Label[/] : [success]value[/]"
)
```

### 3. Info Messages
```python
console.print("[info]Found[/] [warning]82[/] [info]image file(s) in[/] [detail]/path[/][info].[/]")
```

### 4. Success Messages
```python
console.print("[success]✓[/] [success]Operation complete[/]")
```

### 5. Warnings
```python
console.print("[warning]![/] [warning]Warning message[/]")
```

### 6. Errors
```python
console.print("[error]✗[/] [error]Error message[/]")
```

### 7. Progress Bars
```python
from rich.progress import Progress, BarColumn, TimeRemainingColumn

with Progress(
    BarColumn(bar_width=40),
    TimeRemainingColumn(),
) as progress:
    task = progress.add_task("[progress]Processing...[/]", total=100)
    for i in range(100):
        progress.update(task, advance=1)
```

### 8. Tables
```python
from rich.table import Table

table = Table(title="[header]Results[/]")
table.add_column("File", style="detail")
table.add_column("Status", style="success")
table.add_row("file.txt", "[success]✓[/]")
console.print(table)
```

### 9. Prompts / Input
```python
answer = console.input("[info]Proceed? [warning][y/N][/] [info]: [/]")
```

### 10. Summary / Stats
```python
from rich.panel import Panel

summary = Panel(
    "[info]Total:[/] [detail]82[/]\n"
    "[info]Processed:[/] [detail]80[/]\n"
    "[info]Failed:[/] [detail]2[/]",
    title="[header]Summary[/]",
)
console.print(summary)
```

---

## Color Usage Guide

| Tag | Use For | Example |
|-----|---------|---------|
| `[info]` | Standard messages, labels, punctuation | "Processing" text, field names |
| `[detail]` | Values, counts, paths, emphasis | "82", "/dataset", filenames |
| `[success]` | Positive outcomes, checkmarks | ✓, "complete", "enabled" |
| `[warning]` | Numbers, cautions, disabled states | "82", "disabled", "No API key" |
| `[error]` | Failures, critical issues | ✗, "failed", error reasons |
| `[header]` | Section titles, banners | "Analysis Complete" |
| `[progress]` | Progress indicators, active bars | "[progress]Processing...[/]" |
| `[muted]` | Secondary info, de-emphasized | timestamps, minor details |

---

## Accessibility

- **Avoid color-only indicators**: Always pair color with symbols (✓, ✗, !) or text labels
- **Test contrast**: Use `console.print(..., soft_wrap=True)` to preserve line breaks on small terminals
- **No flickering**: Avoid repeated `console.clear()` in tight loops; use `Progress` for animated updates

---

## Example: Minimal CLI

```python
#!/usr/bin/env python3
from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    "info": "white",
    "detail": "bright_white",
    "success": "bright_green",
    "warning": "bright_yellow",
    "error": "bright_red",
    "header": "bright_magenta",
})

console = Console(theme=THEME)

console.rule("[header]My CLI Tool[/]")
console.print("[info]Finding files...[/] [detail]./dataset[/]")
console.print("[success]✓[/] [success]Found[/] [warning]82[/] [info]files[/]")
console.print("[info]Status: [/][success]ready[/]")
```

---

## When to Deviate

- **Explicit user request**: If end-user requests a different theme/library, document it in that project's README
- **Accessibility needs**: High-contrast or monochrome variants (create alternate theme)
- **Textual approval**: If interactive components are needed, escalate to Textual spec

Otherwise: **Use this standard without modification**. This reduces iteration cycles.

---

## Checklist for Code Generation

When requesting a TUI, include:
- [ ] "Use Rich standard from TUI_SPEC.md"
- [ ] Any custom colors/symbols beyond the standard (or omit to use defaults)
- [ ] Output format: headers, tables, progress, summary (specify which)

Example request:
```
Create a CLI tool that:
- Uses Rich standard from TUI_SPEC.md
- Shows a progress bar with file count
- Outputs a summary table at the end
```
