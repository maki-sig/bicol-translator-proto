import argparse
from collections import Counter, deque
from contextlib import nullcontext
import csv
from difflib import get_close_matches
import importlib
import json
from pathlib import Path
import pickle
import re
import shutil
import sqlite3
import sys
from time import perf_counter

from . import data, ui
from .inference import ENGINES, TranslationResult, Translator

PROCESSED = Path("data/processed")
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = Path("data/splits")
ARTIFACTS = Path("artifacts")
TOKENIZER = ARTIFACTS / "tokenizer/bikol_naga_bpe.model"
MODEL = ARTIFACTS / "model/best.pt"
MARIAN_DIR = ARTIFACTS / "marian/opus-mt-en-bcl"
DICTIONARY = Path("data/dictionary/bikol_naga.sqlite")
TRAIN_DICTIONARY = DICTIONARY.with_name("bikol_naga_train_only.sqlite")
METHODS = ("auto", "dictionary", "nmt")
RUNTIME_ERRORS = (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error, pickle.UnpicklingError, EOFError)
TSV_FIELDS = ("source", "translation", "alternatives", "method", "elapsed_ms", "warnings")


class TranslateParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"error: {message}\nhint: Run with --help for accepted options.\n")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def translate_options(parser):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--interactive", action="store_true")
    modes.add_argument("--text")
    modes.add_argument("--input-file", type=Path)
    path(parser, "output-file")
    path(parser, "dictionary", REPO_ROOT / DICTIONARY)
    path(parser, "model", REPO_ROOT / MODEL)
    path(parser, "tokenizer", REPO_ROOT / TOKENIZER)
    path(parser, "marian-dir", REPO_ROOT / MARIAN_DIR)
    parser.add_argument("--engine", choices=ENGINES, default="marian",
                        help="NMT engine to use after the dictionary lookup (default: marian)")
    parser.add_argument("--method", choices=METHODS, default="auto")
    parser.add_argument("--beam-size", type=int, choices=range(1, 51), default=5, metavar="N")
    parser.add_argument("--top-k", type=int, choices=range(1, 51), default=3, metavar="N")
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--quantize", action="store_true", help="experiment with dynamic int8 layers in the custom engine")
    parser.add_argument("--allow-low-confidence", action="store_true",
                        help="include model outputs below the accuracy cutoff")
    parser.add_argument("--json", action="store_true", help="write JSON output")
    parser.add_argument("--json-pretty", action="store_true", help="indent one-shot JSON output")
    parser.add_argument("--plain", action="store_true", help="write ASCII key=value lines")
    parser.add_argument("--quiet", action="store_true", help="print only the primary translation in text mode")
    parser.add_argument("--verbose", "--details", dest="verbose", action="store_true",
                        help="show scores, token counts and model load time")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--no-color", dest="color", action="store_const", const="never")
    parser.add_argument("--ascii", action="store_true", help="use ASCII terminal symbols")
    parser.add_argument("--width", type=positive_int, metavar="N", help="override terminal width")
    parser.add_argument("--format", choices=("tsv", "jsonl", "text"), help="batch output format, default TSV")
    parser.add_argument("--skip-blank", action="store_true", help="skip empty batch lines")
    parser.add_argument("--fail-fast", action="store_true", help="stop a batch on the first runtime error")
    parser.add_argument("--debug", action="store_true", help="include tracebacks in error output")


def path(parser, flag, default=None):
    parser.add_argument("--" + flag, type=Path, default=default)


def main(command):
    if command == "translate":
        return translate_main()
    parser = argparse.ArgumentParser()
    if command == "verify_dataset":
        path(parser, "full", PROCESSED / "bikol_naga_nmt.csv")
        path(parser, "sentences", PROCESSED / "bikol_naga_sentences.csv")
        path(parser, "review", PROCESSED / "bikol_naga_review.csv")
        path(parser, "report", PROCESSED / "verification_report.txt")
        path(parser, "build-report", PROCESSED / "bikol_naga_report.txt")
        path(parser, "verified")
    elif command == "split_dataset":
        path(parser, "input", PROCESSED / "bikol_naga_sentences.csv")
        path(parser, "output-dir", SPLITS)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--ratios", nargs=3, type=float, default=(.8, .1, .1))
        parser.add_argument("--threshold", type=float, default=.9)
    elif command == "preprocess":
        path(parser, "splits-dir", SPLITS)
        path(parser, "output-dir", SPLITS)
    elif command == "train_tokenizer":
        path(parser, "train-en", SPLITS / "train.en.txt")
        path(parser, "train-bk", SPLITS / "train.bk.txt")
        path(parser, "output-dir", TOKENIZER.parent)
        parser.add_argument("--vocab-size", type=int, default=4000)
        parser.add_argument("--character-coverage", type=float, default=.9995)
    elif command == "train_nmt":
        path(parser, "config", Path("configs/cpu_small.json"))
        path(parser, "train", SPLITS / "train.csv")
        path(parser, "val", SPLITS / "val.csv")
        path(parser, "tokenizer", TOKENIZER)
        path(parser, "output-dir", MODEL.parent)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
    elif command == "build_dictionary":
        path(parser, "input", PROCESSED / "bikol_naga_nmt.csv")
        path(parser, "output", DICTIONARY)
    elif command == "calibrate_nmt":
        path(parser, "model", MODEL)
        path(parser, "tokenizer", TOKENIZER)
        path(parser, "train", SPLITS / "train.csv")
        path(parser, "val", SPLITS / "val.csv")
        parser.add_argument("--beam-size", type=int, choices=range(1, 51), default=5, metavar="N")
    elif command == "evaluate_nmt":
        path(parser, "model", MODEL)
        path(parser, "tokenizer", TOKENIZER)
        path(parser, "test", SPLITS / "test.csv")
        path(parser, "dictionary", TRAIN_DICTIONARY)
        path(parser, "production-dictionary", DICTIONARY)
        path(parser, "output-dir", ARTIFACTS / "eval")
        parser.add_argument("--beam-size", type=int, default=5)
        parser.add_argument("--bootstrap-samples", type=int, default=500)
    else:
        raise ValueError(f"Unknown command: {command}")
    args = parser.parse_args()
    try:
        if command == "verify_dataset":
            data.verify(args.full, args.sentences, args.review, args.report, args.verified, args.build_report)
            print(f"Dataset verified. Report: {args.report}")
        elif command == "split_dataset":
            data.split_files(args.input, args.output_dir, args.seed, args.ratios, args.threshold)
        elif command == "preprocess":
            data.preprocess(args.splits_dir, args.output_dir)
        elif command == "build_dictionary":
            from .dictionary import build
            build(args.input, args.output)
        elif command == "train_tokenizer":
            from .tokenizer import train
            train(args.train_en, args.train_bk, args.output_dir, args.vocab_size, args.character_coverage)
        elif command == "train_nmt":
            from .training import train
            config = json.loads(args.config.read_text(encoding="utf-8-sig"))
            train(config, args.train, args.val, args.tokenizer, args.output_dir, args.resume, args.dry_run)
        elif command == "calibrate_nmt":
            from .evaluation import calibrate
            calibrate(args.model, args.tokenizer, args.train, args.val, args.beam_size)
        elif command == "evaluate_nmt":
            from .evaluation import evaluate
            if not 1 <= args.beam_size <= 50:
                raise ValueError("Beam size must be between 1 and 50")
            evaluate(args.model, args.tokenizer, args.test, args.dictionary, args.production_dictionary,
                     args.output_dir, args.beam_size, args.bootstrap_samples)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def terminal_view(args, stream, *, file_output=False):
    return ui.theme(stream, "never" if args.plain else args.color, args.ascii or args.plain,
                    args.width, file_output=file_output)


def translate_main():
    parser = TranslateParser(description="Translate English to Bikol-Naga offline.")
    translate_options(parser)
    ui.configure_windows_streams(sys.stdout, sys.stderr)
    args = parser.parse_args()
    if not args.input_file and (args.output_file or args.format or args.skip_blank or args.fail_fast):
        parser.error("--output-file, --format, --skip-blank and --fail-fast require --input-file")
    try:
        return translate(args)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except RUNTIME_ERRORS as error:
        ui.show_error(error, terminal_view(args, sys.stderr), debug=args.debug)
        return 1


class ConsoleTranslator(Translator):
    def __init__(self, args, dictionary):
        config = {"allow_low_confidence": args.allow_low_confidence, "quantize": args.quantize}
        if args.threads is not None:
            config["threads"] = args.threads
        super().__init__(dictionary, args.model, args.tokenizer, config,
                         engine=args.engine, marian_dir=args.marian_dir)
        self.args = args
        self.load_ms = None
        self.clear_status = None

    def load_neural(self):
        if self.neural is not None:
            return self.neural
        status = ui.is_terminal(sys.stderr) and not self.args.quiet
        if status:
            if self.clear_status is not None:
                self.clear_status()
            print("Loading model...", file=sys.stderr, flush=True)
        start = perf_counter()
        try:
            neural = super().load_neural()
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"Invalid model checkpoint metadata: {error}") from error
        self.load_ms = (perf_counter() - start) * 1000
        if status and self.args.verbose:
            print(f"Model loaded in {self.load_ms / 1000:.1f} s", file=sys.stderr)
        return neural

    def request(self, source):
        self.load_ms = None
        return self.translate(source, self.args.method, self.args.top_k, self.args.beam_size,
                              strict_errors=True)

    def set_low_confidence(self, allowed):
        self.config["allow_low_confidence"] = allowed
        if self.neural is not None:
            self.neural.settings["allow_low_confidence"] = allowed


def text_output(args):
    return "quiet" if args.quiet else "plain" if args.plain else "human"


def translate(args):
    session = args.text is None and args.input_file is None
    dictionary = args.dictionary if session or args.method != "nmt" else None
    translator = ConsoleTranslator(args, dictionary)
    try:
        if args.text is not None:
            result = translator.request(args.text)
            if args.json or args.json_pretty:
                print(ui.result_json(result, pretty=args.json_pretty))
            else:
                ui.render(result, terminal_view(args, sys.stdout), output=text_output(args),
                          details=args.verbose, load_ms=translator.load_ms)
            return 0
        if args.input_file:
            return batch(translator, args)
        return interactive(translator, args)
    finally:
        translator.close()


def batch(translator, args):
    sources = args.input_file.read_text(encoding="utf-8-sig").splitlines()
    if args.skip_blank:
        sources = [source for source in sources if source.strip()]
    output = args.output_file
    if output:
        if output.resolve() == args.input_file.resolve() or output.exists() and output.samefile(args.input_file):
            raise ValueError("Batch output must differ from input")
        output.parent.mkdir(parents=True, exist_ok=True)
    if args.format:
        format_name = args.format
    elif not output and (args.json or args.json_pretty):
        format_name = "jsonl"
    elif not output and (args.plain or args.quiet):
        format_name = "text"
    else:
        format_name = "tsv"
    stream_context = output.open("w", encoding="utf-8", newline="") if output else nullcontext(sys.stdout)
    with stream_context as stream:
        return batch_rows(translator, sources, args, stream, format_name)


def batch_rows(translator, sources, args, stream, format_name):
    output_view = terminal_view(args, stream, file_output=bool(args.output_file))
    status_view = terminal_view(args, sys.stderr)
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n") if format_name == "tsv" else None
    if writer:
        writer.writerow(TSV_FIELDS)
    counts = Counter()
    start = perf_counter()
    progress = ui.is_terminal(sys.stderr) and not args.quiet
    progress_width = 0
    last_update = start - 1

    def clear_progress():
        nonlocal progress_width
        if progress_width:
            print("\r" + " " * progress_width + "\r", end="", file=sys.stderr, flush=True)
            progress_width = 0

    if progress and isinstance(translator, ConsoleTranslator):
        translator.clear_status = clear_progress
    try:
        for index, source in enumerate(sources, 1):
            row_start = perf_counter()
            runtime_error = False
            try:
                result = translator.request(source)
            except RUNTIME_ERRORS as error:
                clear_progress()
                runtime_error = True
                counts["errors"] += 1
                ui.show_error(error, status_view, debug=args.debug)
                result = TranslationResult(source, [], "no_match", (perf_counter() - row_start) * 1000,
                                           [str(error)])
            counts[result.method] += 1
            if writer:
                first = result.translations[0].text if result.translations else ""
                alternatives = [ui.candidate_object(candidate, legacy=True) for candidate in result.translations]
                writer.writerow((source, first, json.dumps(alternatives, ensure_ascii=False), result.method,
                                 result.elapsed_ms, " | ".join(result.warnings)))
            elif format_name == "jsonl":
                print(ui.result_json(result), file=stream)
            else:
                if index > 1 and not args.quiet:
                    print(file=stream)
                ui.render(result, output_view, output=text_output(args), compact=True, details=args.verbose,
                          load_ms=translator.load_ms, position=(index, len(sources)))
            if runtime_error and args.fail_fast:
                break
            now = perf_counter()
            if progress and (now - last_update >= .1 or index == len(sources)):
                sep = status_view.separator
                message = sep.join((f"Translating {index}/{len(sources)}",
                                    f"{index / len(sources):.0%}", f"{now - start:.1f} s"))
                print("\r" + message.ljust(progress_width), end="", file=sys.stderr, flush=True)
                progress_width = len(message)
                last_update = now
    finally:
        clear_progress()
        if isinstance(translator, ConsoleTranslator):
            translator.clear_status = None
    if not args.quiet:
        total = sum(counts[method] for method in ("dictionary", "nmt", "no_match"))
        parts = ["Stopped" if total < len(sources) else "Done", f"{total} rows",
                 f"dictionary {counts['dictionary']}", f"nmt {counts['nmt']}",
                 f"no match {counts['no_match']}"]
        if counts["errors"]:
            parts.append(f"errors {counts['errors']}")
        parts.append(f"{perf_counter() - start:.1f} s")
        print(status_view.separator.join(parts), file=sys.stderr)
        if args.output_file:
            print(f"Output: {args.output_file}", file=sys.stderr)
    return int(bool(counts["errors"]))


HELP = {
    "Translation": {
        "method": ("auto|dictionary|nmt", "Set the translation method (dict and d also work)."),
        "engine": ("custom|marian", "Select the NMT engine used after a dictionary miss."),
        "beam": ("1-50", "Set the beam size."),
        "topk": ("1-50", "Set the maximum number of translations."),
        "low": ("on|off", "Include or hide low-confidence guesses."),
    },
    "Display": {
        "timing": ("on|off", "Show or hide elapsed time."),
        "color": ("on|off", "Enable or disable terminal color."),
        "json": ("on|off", "Emit one JSON object per translation."),
        "clear": ("", "Clear the terminal screen."),
    },
    "Session": {
        "config": ("", "Show active settings and artifact paths."),
        "model": ("", "Show checkpoint metadata and confidence cutoff."),
        "stats": ("", "Show translation counts and mean elapsed time."),
        "history": ("", "Show the last 20 inputs in this session."),
        "help": ("[command]", "Show all commands or help for one command."),
        "quit": ("", "Exit the shell. Aliases: :q and :exit."),
    },
}
COMMANDS = {command: detail for group in HELP.values() for command, detail in group.items()}
COMMAND_ALIASES = {"q": "quit", "exit": "quit"}


def command_help(view, value=""):
    if value:
        name = value.lstrip(":").lower()
        name = COMMAND_ALIASES.get(name, name)
        if name in COMMANDS:
            usage, detail = COMMANDS[name]
            ui.wrapped(view, f":{name} {usage}".rstrip(), detail, label_width=27)
        else:
            show_unknown_command(view, name)
        return
    for section, commands in HELP.items():
        print(view.style(section, "translation"), file=view.stream)
        for name, (usage, detail) in commands.items():
            instruction = f":{name} {usage}".rstrip()
            ui.wrapped(view, instruction, detail, label_width=31)


def show_unknown_command(view, name):
    similar = get_close_matches(name, COMMANDS, n=1)
    hint = f"Did you mean :{similar[0]}?" if similar else "Enter :help for commands."
    print(f"Unknown command :{ui.human_text(name)}. {hint}", file=view.stream)


def on_off(value):
    return "on" if value else "off"


def session_path(path):
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def session_config(view, args):
    for label, value in (
        ("Method", args.method), ("Engine", args.engine), ("Beam", args.beam_size), ("Top-k", args.top_k),
        ("Timing", on_off(args.timing)), ("Low conf.", on_off(args.allow_low_confidence)),
        ("Color", on_off(view.color)), ("JSON", on_off(args.json_mode)), ("Dictionary", session_path(args.dictionary)),
        ("Model", session_path(args.model)), ("Tokenizer", session_path(args.tokenizer)),
        ("MarianMT", session_path(args.marian_dir)),
        ("Threads", args.threads or "auto"),
    ):
        ui.wrapped(view, label, value)


def session_model(view, translator):
    neural = translator.load_neural()
    if translator.engine == "marian":
        from .marian import MODEL_ID
        ui.wrapped(view, "Engine", MODEL_ID)
        ui.wrapped(view, "Snapshot", session_path(translator.args.marian_dir))
        ui.wrapped(view, "Parameters", f"{neural.parameter_count:,}")
        return
    for label, value in (
        ("Checkpoint", session_path(translator.args.model)), ("Epoch", neural.checkpoint_epoch or "unknown"),
        ("Parameters", f"{neural.parameter_count:,}"),
        ("Train hash", neural.training_hash), ("Val hash", neural.validation_hash),
        ("NLL cutoff", neural.max_token_nll if neural.max_token_nll is not None else "unavailable"),
    ):
        ui.wrapped(view, label, value)


def session_stats(view, counts, elapsed_ms):
    total = sum(counts.values())
    for label, value in (("Dictionary", counts["dictionary"]), ("NMT", counts["nmt"]),
                         ("No match", counts["no_match"]), ("Mean time", f"{elapsed_ms / total:.1f} ms" if total else "--")):
        ui.wrapped(view, label, value)


def clear_terminal(view):
    if ui.supports_ansi(view.stream):
        print("\033[2J\033[H", end="", file=view.stream, flush=True)
    elif ui.is_terminal(view.stream):
        print("\n" * shutil.get_terminal_size((100, 24)).lines, end="", file=view.stream, flush=True)


def session_command(text, view, translator, args, counts, elapsed_ms, history):
    parts = re.split(r"\s*=\s*|\s+", text[1:].strip(), maxsplit=1)
    name = parts[0].lower() or "help"
    value = parts[1].strip().lower() if len(parts) == 2 else ""
    name = COMMAND_ALIASES.get(name, name)
    if name not in COMMANDS:
        show_unknown_command(view, name)
        return False
    if name == "quit":
        return True
    if name == "help":
        command_help(view, value)
    elif name == "method":
        method = "dictionary" if value in ("d", "dict") else value
        if method in METHODS:
            args.method = method
            print(f"Method set to {method}.", file=view.stream)
        else:
            print("Use :method auto, dictionary or nmt.", file=view.stream)
    elif name == "engine":
        if value in ENGINES:
            translator.select_engine(value)
            args.engine = value
            print(f"NMT engine set to {value}.", file=view.stream)
        else:
            print("Use :engine custom or marian.", file=view.stream)
    elif name in ("beam", "topk"):
        if not re.fullmatch(r"[0-9]{1,2}", value) or not 1 <= int(value) <= 50:
            print(f"Use :{name} with a number from 1 to 50.", file=view.stream)
        else:
            setattr(args, "beam_size" if name == "beam" else "top_k", int(value))
            label = "Beam" if name == "beam" else "Top-k"
            print(f"{label} set to {int(value)}.", file=view.stream)
    elif name in ("timing", "low", "color", "json"):
        if value not in ("on", "off"):
            print(f"Use :{name} on or off.", file=view.stream)
        else:
            attr = {"timing": "timing", "low": "allow_low_confidence", "color": "color", "json": "json_mode"}[name]
            setting = ("always" if value == "on" else "never") if name == "color" else value == "on"
            setattr(args, attr, setting)
            if name == "low":
                translator.set_low_confidence(setting)
            label = {"timing": "Timing", "low": "Low confidence", "color": "Color", "json": "JSON"}[name]
            print(f"{label} set to {value}.", file=view.stream)
    elif name == "config":
        session_config(view, args)
    elif name == "model":
        session_model(view, translator)
    elif name == "stats":
        session_stats(view, counts, elapsed_ms)
    elif name == "history":
        if not history:
            print("No inputs in this session yet.", file=view.stream)
        for index, source in enumerate(history, 1):
            ui.wrapped(view, str(index), source)
    elif name == "clear":
        clear_terminal(view)
    return False


def interactive(translator, args):
    args.timing = True
    args.json_mode = args.json or args.json_pretty
    counts, history = Counter(), deque(maxlen=20)
    elapsed_ms = 0.
    status = 0
    live = ui.is_terminal(sys.stdin) and ui.is_terminal(sys.stderr)
    if live:
        try:
            importlib.import_module("readline")
        except (ImportError, OSError, RuntimeError):
            pass
        view = terminal_view(args, sys.stderr)
        sep = view.separator
        ui.wrapped(view, "", f"Bikol-Naga Translator{sep}offline", label_width=0, role="translation")
        engine_name = "MarianMT" if args.engine == "marian" else args.model.name
        ui.wrapped(view, "", sep.join(("Dictionary + Transformer", engine_name,
                                      f"beam {args.beam_size}", f"top-k {args.top_k}")), label_width=0)
        ui.wrapped(view, "", f"Type English. :help for commands{sep}:q to quit.", label_width=0)
    while True:
        view = terminal_view(args, sys.stderr)
        if live:
            method = "dict" if args.method == "dictionary" else args.method
            marker = "›" if view.unicode else ">"
            print(f"{method} {marker} ", end="", file=sys.stderr, flush=True)
        try:
            source = input()
        except EOFError:
            if live:
                print(file=sys.stderr)
            return status
        text = source.strip()
        if not text:
            print("Type an English word or sentence. :help for commands.", file=sys.stderr)
            continue
        try:
            if text.startswith(":"):
                if session_command(text, view, translator, args, counts, elapsed_ms, history):
                    return status
                continue
            history.append(source)
            result = translator.request(source)
            counts[result.method] += 1
            elapsed_ms += result.elapsed_ms
            if args.json_mode:
                print(ui.result_json(result))
            else:
                ui.render(result, terminal_view(args, sys.stdout), output=text_output(args),
                          compact=True, timing=args.timing, details=args.verbose, load_ms=translator.load_ms)
        except RUNTIME_ERRORS as error:
            ui.show_error(error, view, debug=args.debug)
            status = 1
