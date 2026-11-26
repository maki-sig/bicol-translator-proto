"""Terminal capabilities and output for the offline translator."""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
import re
import shutil
import sys
import textwrap
import traceback
from typing import TextIO


STYLES = {
    "label": "2", "translation": "1;36", "index": "2", "dictionary": "32",
    "nmt": "36", "no_match": "33", "note": "33", "error": "1;31",
    "hint": "2", "high": "32", "medium": "33", "low": "31",
    "below cutoff": "31",
}
CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def is_terminal(stream):
    return stream.isatty()


def configure_windows_streams(*streams):
    if os.name == "nt":
        for stream in streams:
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


@lru_cache(maxsize=8)
def enable_windows_vt(fd):
    try:
        import ctypes
        from ctypes import wintypes
        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        handle = msvcrt.get_osfhandle(fd)
        mode = wintypes.DWORD()
        return bool(kernel.GetConsoleMode(handle, ctypes.byref(mode))
                    and kernel.SetConsoleMode(handle, mode.value | 0x0004))
    except (OSError, ValueError):
        return False


def supports_ansi(stream):
    if not is_terminal(stream):
        return False
    if os.name != "nt":
        return os.environ.get("TERM") != "dumb"
    try:
        return enable_windows_vt(stream.fileno())
    except (AttributeError, OSError, ValueError):
        return False


@dataclass(frozen=True)
class Theme:
    color: bool
    unicode: bool
    width: int
    stream: TextIO

    @property
    def separator(self):
        return " · " if self.unicode else " - "

    def style(self, value, role):
        return f"\033[{STYLES[role]}m{value}\033[0m" if self.color and role else value


def theme(stream, color="auto", ascii_only=False, width=None, *, file_output=False):
    tty = is_terminal(stream)
    color_on = not file_output and (
        (color == "always" and (not tty or os.name != "nt" or supports_ansi(stream)))
        or (color == "auto" and "NO_COLOR" not in os.environ and supports_ansi(stream))
    )
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "→›·—".encode(encoding)
        unicode_on = not ascii_only
    except (UnicodeError, LookupError):
        unicode_on = False
    return Theme(color_on, unicode_on, width or shutil.get_terminal_size((100, 24)).columns, stream)


def candidate_object(candidate, *, legacy=False):
    value = {key: getattr(candidate, key) for key in ("text", "score", "source", "category")}
    if not legacy:
        for key in ("nll", "confidence", "engine"):
            if (number := getattr(candidate, key)) is not None:
                value[key] = number
    return value


def result_json(result, *, pretty=False):
    return json.dumps({
        "source_text": result.source_text,
        "translations": [candidate_object(candidate) for candidate in result.translations],
        "method": result.method,
        "elapsed_ms": result.elapsed_ms,
        "warnings": result.warnings,
    }, ensure_ascii=False, indent=2 if pretty else None)


def human_text(value):
    return CONTROLS.sub(lambda match: f"\\x{ord(match[0]):02x}", str(value))


def wrapped(view, label, value, *, compact=False, role="", prefix="", prefix_role="", label_width=None):
    padding = "  " if compact else ""
    if label_width is None:
        label_width = max(9 if compact else 11, len(label) + 1)
    stacked = view.width - label_width - len(padding) < 40
    if stacked:
        indent = padding + "  "
        if label:
            print(padding + view.style(label, "label"), file=view.stream)
        start = indent
    else:
        indent = padding + " " * label_width
        start = padding + view.style(f"{label:<{label_width}}", "label")
    value = human_text(value)
    # Wrap before adding ANSI codes so styling never changes column widths.
    lines = textwrap.wrap(prefix + value, max(1, view.width - len(indent)),
                          subsequent_indent=" " * len(prefix), break_long_words=False,
                          break_on_hyphens=False) or [""]
    first = lines[0]
    styled = view.style(first, role)
    if prefix:
        styled = view.style(first[:len(prefix)], prefix_role) + view.style(first[len(prefix):], role)
    print(start + styled, file=view.stream)
    for line in lines[1:]:
        print(indent + view.style(line, role), file=view.stream)


def confidence_label(candidate):
    if candidate.below_cutoff:
        return "below cutoff"
    if candidate.confidence is None:
        return ""
    if candidate.confidence >= .66:
        return "high"
    return "medium" if candidate.confidence >= .33 else "low"


def plain_value(value):
    return human_text(value).replace("\n", r"\n").replace("\t", r"\t").encode(
        "ascii", errors="backslashreplace").decode("ascii")


def guidance(result, *, interactive):
    commands = ":low on (or --allow-low-confidence)" if interactive else "--allow-low-confidence"
    hints, reasons = [], []
    for warning in result.warnings:
        if warning.startswith("Use --allow-low-confidence"):
            hints.append(f"Use {commands} to inspect model guesses.")
        elif "all translations below the cutoff" in warning:
            reasons.append(warning.split(" Use ", 1)[0])
            hints.append(f"Use {commands} to inspect model guesses.")
        else:
            reasons.append(warning)
    if result.method == "no_match" and not hints:
        hints.append("Enter an English word or sentence." if not result.source_text.strip()
                     else "Try a shorter sentence or rephrase the English.")
    if result.method == "no_match" and not reasons:
        reasons.append("No translation found.")
    return reasons, hints


def render(result, view, *, output="human", compact=False, timing=True,
           details=False, load_ms=None, position=None):
    candidates = result.translations
    primary = candidates[0] if candidates else None
    if output == "quiet":
        if primary:
            print(primary.text, file=view.stream)
        return
    seen = {primary.text} if primary else set()
    alternatives = []
    for candidate in candidates[1:]:
        if candidate.text not in seen:
            alternatives.append(candidate)
            seen.add(candidate.text)
    if output == "plain":
        rows = [("source", result.source_text), ("translation", primary.text if primary else "")]
        rows.extend(("alternative", alternative.text) for alternative in alternatives)
        rows.extend((("method", result.method), ("category", (primary.category or "") if primary else "")))
        if timing:
            rows.append(("elapsed_ms", f"{result.elapsed_ms:.1f}"))
        rows.extend(("warning", warning) for warning in result.warnings or [""])
        for key, value in rows:
            print(f"{key}={plain_value(value)}", file=view.stream)
        return

    def line(label, value, **options):
        wrapped(view, label, value, compact=compact, **options)

    if position:
        number, total = position
        line(f"[{number}/{total}]", result.source_text)
    elif not compact:
        line("English", result.source_text)
    line("Bikol", primary.text if primary else ("—" if view.unicode else "--"), role="translation")
    for i, candidate in enumerate(alternatives, 1):
        line("Also" if i == 1 else "", candidate.text,
             prefix=f"{i}. " if len(alternatives) > 1 else "", prefix_role="index")
    method = "dict" if compact and result.method == "dictionary" and not position else result.method
    badge = confidence_label(primary) if primary else ""
    metadata = [method]
    if primary and primary.engine:
        metadata.append("MarianMT" if primary.engine == "marian" else "custom Transformer")
    if primary and primary.category:
        metadata.append(primary.category)
    if badge:
        metadata.append(badge if badge == "below cutoff" else f"{badge} confidence")
    if timing and compact:
        metadata.append(f"{result.elapsed_ms:.1f} ms")
    line("" if compact and not position else "Method", view.separator.join(metadata),
         role=badge or result.method)
    reasons, hints = guidance(result, interactive=compact and not position)
    for reason in reasons:
        line("Reason" if result.method == "no_match" else "Note", reason, role="note")
    for hint in hints:
        line("Hint", hint, role="hint")
    if timing and not compact:
        line("Time", f"{result.elapsed_ms:.1f} ms")
    if details:
        line("Details", f"candidates={len(candidates)}")
        for i, candidate in enumerate(candidates, 1):
            metrics = []
            for name in ("score", "nll", "confidence"):
                if (value := getattr(candidate, name)) is not None:
                    metrics.append(f"{name}={value:.3f}")
            if candidate.token_count is not None:
                metrics.append(f"tokens={candidate.token_count}")
            if metrics:
                line("", " ".join(metrics), prefix=f"{i}. ", prefix_role="index")
        if load_ms is not None:
            line("Model load", f"{load_ms:.1f} ms")


def error_hint(error):
    message = str(error).lower()
    if "token limit" in message:
        return "Split the English into shorter sentences and translate each one."
    if "requirements-marian.txt" in message:
        return "Install the optional dependencies from requirements-marian.txt."
    if "marianmt files missing" in message or "marianmt model directory" in message:
        return "Run scripts/download_marian.py or pass --marian-dir with the local snapshot."
    if "tokenizer" in message and "do not match" in message:
        return "Pass --model and --tokenizer for artifacts from the same training run."
    if "dictionary" in message or "database" in message or "no such table" in message:
        return "Run scripts/build_dictionary.py or pass --dictionary with a readable SQLite file."
    if "tokenizer" in message:
        return "Run scripts/train_tokenizer.py or pass --tokenizer with a readable tokenizer file."
    if "checkpoint" in message or "model" in message:
        return "Run scripts/train_nmt.py or pass --model with a readable checkpoint."
    if isinstance(error, OSError):
        return "Check that the input paths exist and that the output location is writable."
    return "Check the input and options. Run with --help for usage."


def show_error(error, view, *, debug=False):
    print(f"{view.style('error:', 'error')} {human_text(error)}", file=view.stream)
    print(f"{view.style('hint:', 'hint')} {error_hint(error)}", file=view.stream)
    if debug:
        traceback.print_exception(error, file=view.stream)
