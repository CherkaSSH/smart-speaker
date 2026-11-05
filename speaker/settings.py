"""Terminal editor for config.toml: `speaker settings`.

Edits go through tomlkit, so comments and layout survive saving. Each field
shows the comment written above it in the file as its help text.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (ContentSwitcher, Footer, Header, Input, Label, ListItem, ListView, Select, Static,
                             Switch, TextArea)
from tomlkit.items import Comment, InlineTable, String, StringType, Whitespace

# Which processes read each section, i.e. what must restart after editing it.
USERS = {
    "character": ("brain", "voice"), "wake": ("voice",), "gate": ("voice",), "audio": ("voice", "brain"),
    "music": ("brain", "voice"),
    "asr": ("asr", "voice"), "tts": ("voice",), "llm": ("llm", "brain"), "remote_llm": ("brain",),
    "omp": ("brain",), "web_search": ("brain",), "news": ("brain",), "shopping": ("brain",),
    "memory": ("embed", "brain"), "brain": ("brain",), "telegram": ("brain",), "discord": ("brain",),
    "voice": ("voice",), "speakers": ("voice",),
}
CHOICES = {
    ("asr", "device"): ["vulkan", "cpu"],
    ("tts", "ru_speaker"): ["aidar", "baya", "kseniya", "xenia", "eugene"],
    ("web_search", "mode"): ["turbo", "fast", "basic", "advanced"],
    ("music", "after_request"): ["ask", "continue", "stop"],
}
RANGES = {
    ("wake", "threshold"): (0, 1), ("gate", "vad_threshold"): (0, 1), ("audio", "volume"): (0, 10),
    ("memory", "recall_threshold"): (0, 1), ("llm", "temperature"): (0, 2), ("remote_llm", "temperature"): (0, 2),
    ("brain", "history_minutes"): (0, 1_440), ("audio", "night_volume"): (0, 10),
    ("speakers", "threshold"): (0, 1), ("speakers", "min_seconds"): (0, 10),
}


@dataclass
class Field:
    section: str
    key: str
    kind: str  # bool, int, float, str, text, list_str, list_int, table
    help: str
    value: object  # plain Python value as loaded

    @property
    def id(self) -> str:
        return f"f-{self.section}-{self.key}"


def _kind(item) -> str:
    v = item.unwrap() if hasattr(item, "unwrap") else item
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(item, String):
        return "text" if item.type in (StringType.MLB, StringType.MLL) or "\n" in v else "str"
    if isinstance(item, InlineTable) and all(isinstance(x, str) for x in v.values()):
        return "table"
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return "list_str"
    if isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        return "list_int"
    raise ValueError(f"unsupported value {v!r}")


def load_fields(doc: tomlkit.TOMLDocument) -> dict[str, list[Field]]:
    """Fields per section. A comment block becomes the help of the first key after it;
    the keys it also covers follow directly below, so it isn't repeated."""
    sections: dict[str, list[Field]] = {}
    for name, table in doc.items():
        if not hasattr(table, "value") or not hasattr(table.value, "body"):
            continue
        fields, comment = [], []
        for key, item in table.value.body:
            if isinstance(item, Comment):
                comment.append(item.as_string().strip().lstrip("#").strip())
            elif isinstance(item, Whitespace):
                comment = []  # a blank line detaches the comment above it
            elif key is not None:
                fields.append(Field(name, key.key, _kind(item), " ".join(comment), item.unwrap()))
                comment = []
        sections[name] = fields
    return sections


def parse(field: Field, raw) -> object:
    """Turn widget content into a typed value; ValueError explains what is wrong."""
    k = field.kind
    if k == "bool":
        return bool(raw)
    if k in ("int", "float"):
        text = str(raw).strip()
        try:
            value = int(text) if k == "int" else float(text)
        except ValueError:
            raise ValueError(f"needs {'a whole number' if k == 'int' else 'a number'}") from None
        lo, hi = RANGES.get((field.section, field.key), (None, None))
        if lo is not None and not lo <= value <= hi:
            raise ValueError(f"must be between {lo} and {hi}")
        return value
    if k in ("str", "text"):
        return raw
    if k == "list_str":
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if k == "list_int":
        try:
            return [int(x) for x in str(raw).replace(",", " ").split()]
        except ValueError:
            raise ValueError("needs whole numbers separated by commas") from None
    if k == "table":
        return dict(raw)
    raise ValueError(k)


class SettingsApp(App):
    TITLE = "Smart speaker settings"
    CSS = """
    #sections { width: 22; border-right: solid $panel; }
    .field { height: auto; margin: 0 1 1 1; }
    .name { text-style: bold; }
    .name.dirty { color: $warning; }
    .help { color: $text-muted; }
    .error { color: $error; }
    .field TextArea { height: 6; }
    .field TextArea.long { height: 12; }
    .sub { height: auto; }
    .sub Label { width: 6; padding: 1 0 0 0; }
    .sub Input { width: 1fr; }
    #status { height: 1; padding: 0 1; background: $panel; }
    """
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("ctrl+r", "revert", "Revert"),
        Binding("ctrl+q", "quit_safely", "Quit", priority=True),
    ]

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.doc = tomlkit.parse(path.read_text())
        self.sections = load_fields(self.doc)
        self.fields = {f.id: f for fs in self.sections.values() for f in fs}
        self.pending: dict[str, object] = {}  # field id -> new valid value
        self.errors: dict[str, str] = {}
        self._quit_armed = False

    # Layout -----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ListView(*[ListItem(Label(name), id=f"s-{name}") for name in self.sections], id="sections")
            with ContentSwitcher(initial=f"p-{next(iter(self.sections))}"):
                for name, fields in self.sections.items():
                    with VerticalScroll(id=f"p-{name}"):
                        restart = ", ".join(USERS.get(name, ()))
                        yield Static(f"[b]\\[{name}][/b]  used by: {restart or '—'}", classes="help")
                        for f in fields:
                            yield from self._field(f)
        yield Static("", id="status")
        yield Footer()

    def _field(self, f: Field) -> ComposeResult:
        with Vertical(classes="field"):
            if f.help:  # like in the file: the comment heads the keys it describes
                yield Static(f.help, classes="help")
            hint = "  (one item per line)" if f.kind == "list_str" else ""
            yield Label(f.key + hint, classes="name", id=f"{f.id}-name")
            if f.kind == "bool":
                yield Switch(value=f.value, id=f.id)
            elif (f.section, f.key) in CHOICES:
                options = CHOICES[(f.section, f.key)]
                options = options if f.value in options else [f.value, *options]
                yield Select([(o, o) for o in options], value=f.value, allow_blank=False, id=f.id)
            elif f.kind in ("int", "float", "str"):
                yield Input(str(f.value), id=f.id)
            elif f.kind == "list_int":
                yield Input(", ".join(str(x) for x in f.value), id=f.id)
            elif f.kind in ("text", "list_str"):
                text = f.value.strip("\n") if f.kind == "text" else "\n".join(f.value)
                yield TextArea(text, id=f.id, classes="long" if f.kind == "text" else "")
            elif f.kind == "table":
                with Vertical(id=f.id):
                    for sub, v in f.value.items():
                        with Horizontal(classes="sub"):
                            yield Label(sub)
                            yield Input(v, id=f"{f.id}--{sub}")
            error = Static("", classes="error", id=f"{f.id}-error")
            error.display = False  # shown only while the field is invalid
            yield error

    def on_mount(self) -> None:
        self.sub_title = str(self.path)
        self._status("Ctrl+S save · Ctrl+R revert · Ctrl+Q quit")

    # Editing ----------------------------------------------------------------
    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is not None:
            self.query_one(ContentSwitcher).current = "p-" + event.item.id.removeprefix("s-")

    def _raw(self, f: Field):
        if f.kind == "table":
            return {sub: self.query_one(f"#{f.id}--{sub}", Input).value for sub in f.value}
        widget = self.query_one(f"#{f.id}")
        if isinstance(widget, TextArea):
            text = widget.text
            # Multi-line values are stored with a final newline: `"""` sits on its own line.
            return text.strip("\n") + "\n" if f.kind == "text" else text
        return widget.value

    def _changed(self, widget_id: str | None) -> None:
        if not widget_id:
            return
        fid = widget_id.split("--")[0]
        f = self.fields.get(fid)
        if f is None:
            return
        error = self.query_one(f"#{fid}-error", Static)
        try:
            value = parse(f, self._raw(f))
        except ValueError as e:
            self.errors[fid] = str(e)
            self.pending.pop(fid, None)
            error.update(str(e))
            error.display = True
        else:
            self.errors.pop(fid, None)
            error.display = False
            if value == f.value:
                self.pending.pop(fid, None)
            else:
                self.pending[fid] = value
        self.query_one(f"#{fid}-name", Label).set_class(fid in self.pending or fid in self.errors, "dirty")
        self._quit_armed = False
        self._status(self._summary())

    def on_input_changed(self, event: Input.Changed) -> None:
        self._changed(event.input.id)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        self._changed(event.switch.id)

    def on_select_changed(self, event: Select.Changed) -> None:
        self._changed(event.select.id)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._changed(event.text_area.id)

    def _summary(self) -> str:
        if self.errors:
            return f"{len(self.errors)} invalid field(s): " + ", ".join(
                self.fields[i].section + "." + self.fields[i].key for i in self.errors)
        if self.pending:
            return f"{len(self.pending)} unsaved change(s) — Ctrl+S to save"
        return "No changes"

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    # Saving -----------------------------------------------------------------
    def action_save(self) -> None:
        if self.errors:
            self._status("Fix invalid fields first: " + self._summary())
            return
        if not self.pending:
            self._status("No changes to save")
            return
        restart: set[str] = set()
        for fid, value in self.pending.items():
            f = self.fields[fid]
            table = self.doc[f.section]
            if f.kind == "text":
                # `"""` then a newline, which TOML drops: both delimiters stay on their
                # own lines and the value is unchanged.
                quoted = tomlkit.string(value, multiline=True).as_string()
                table[f.key] = tomlkit.parse(f"x = {quoted[:3]}\n{quoted[3:]}")["x"]
            elif f.kind == "table":
                for sub, v in value.items():
                    table[f.key][sub] = v  # keeps the inline table
            else:
                table[f.key] = value
            f.value = value
            restart.update(USERS.get(f.section, ()))
            self.query_one(f"#{fid}-name", Label).remove_class("dirty")
        tmp = self.path.with_suffix(".toml.tmp")
        tmp.write_text(tomlkit.dumps(self.doc))
        os.replace(tmp, self.path)
        saved = len(self.pending)
        self.pending.clear()
        order = [p for p in ("llm", "embed", "asr", "brain", "voice") if p in restart]
        self._status(f"Saved {saved} change(s). Restart to apply: {', '.join(order)} (or restart `speaker up`)")

    def action_revert(self) -> None:
        for fid in list(self.pending) + list(self.errors):
            f = self.fields[fid]
            if f.kind == "bool":
                self.query_one(f"#{fid}", Switch).value = f.value
            elif (f.section, f.key) in CHOICES:
                self.query_one(f"#{fid}", Select).value = f.value
            elif f.kind in ("int", "float", "str"):
                self.query_one(f"#{fid}", Input).value = str(f.value)
            elif f.kind == "list_int":
                self.query_one(f"#{fid}", Input).value = ", ".join(str(x) for x in f.value)
            elif f.kind in ("text", "list_str"):
                text = f.value.strip("\n") if f.kind == "text" else "\n".join(f.value)
                self.query_one(f"#{fid}", TextArea).text = text
            elif f.kind == "table":
                for sub, v in f.value.items():
                    self.query_one(f"#{fid}--{sub}", Input).value = v
        self._status("Reverted unsaved changes")

    def action_quit_safely(self) -> None:
        if (self.pending or self.errors) and not self._quit_armed:
            self._quit_armed = True
            self._status("Unsaved changes — Ctrl+Q again to quit without saving, Ctrl+S to save")
            return
        self.exit()


def run(path: Path) -> None:
    SettingsApp(path).run()
