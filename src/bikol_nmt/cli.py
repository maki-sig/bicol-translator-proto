import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

from . import data
from .inference import Translator

PROCESSED = Path("data/processed")
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = Path("data/splits")
ARTIFACTS = Path("artifacts")
TOKENIZER = ARTIFACTS / "tokenizer/bikol_naga_bpe.model"
MODEL = ARTIFACTS / "model/best.pt"
DICTIONARY = Path("data/dictionary/bikol_naga.sqlite")
TRAIN_DICTIONARY = DICTIONARY.with_name("bikol_naga_train_only.sqlite")
METHODS = ("auto", "dictionary", "nmt")


def path(parser, flag, default=None):
    parser.add_argument("--" + flag, type=Path, default=default)


def main(command):
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
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument("--interactive", action="store_true")
        modes.add_argument("--text")
        modes.add_argument("--input-file", type=Path)
        path(parser, "output-file")
        path(parser, "dictionary", REPO_ROOT / DICTIONARY)
        path(parser, "model", REPO_ROOT / MODEL)
        path(parser, "tokenizer", REPO_ROOT / TOKENIZER)
        parser.add_argument("--method", choices=METHODS, default="auto")
        parser.add_argument("--beam-size", type=int, choices=range(1, 51), default=5, metavar="N")
        parser.add_argument("--top-k", type=int, choices=range(1, 51), default=3, metavar="N")
        parser.add_argument("--threads", type=int)
        parser.add_argument("--quantize", action="store_true", help="experiment with dynamic int8 linear layers")
        parser.add_argument("--allow-low-confidence", action="store_true",
                            help="include model outputs below the accuracy cutoff")
        parser.add_argument("--json", action="store_true")
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
        else:
            translate(args)
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


def show(result, json_output, timing=True):
    if json_output:
        print(json.dumps(asdict(result), ensure_ascii=False))
        return
    if result.translations:
        print(f"Bikol-Naga: {result.translations[0].text}")
        print("Alternatives:")
        for i, candidate in enumerate(result.translations, 1):
            detail = candidate.category if candidate.score is None else f"score={candidate.score:.3f}"
            print(f"  {i}. {candidate.text} ({candidate.source}, {detail})")
    for warning in result.warnings:
        print(warning)
    print(f"Method: {result.method}")
    if timing:
        print(f"Time: {result.elapsed_ms:.1f} ms")


def translate(args):
    if args.output_file and not args.input_file:
        raise ValueError("--output-file requires --input-file")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive")
    config = {"threads": args.threads} if args.threads is not None else {}
    if args.quantize:
        config["quantize"] = True
    if args.allow_low_confidence:
        config["allow_low_confidence"] = True
    dictionary = None if args.method == "nmt" and (args.text is not None or args.input_file) else args.dictionary
    translator = Translator(dictionary, args.model, args.tokenizer, config)
    try:
        if args.text is not None:
            show(translator.translate(args.text, args.method, args.top_k, args.beam_size), args.json)
        elif args.input_file:
            sources = args.input_file.read_text(encoding="utf-8-sig").splitlines()
            if args.output_file:
                if args.output_file.resolve() == args.input_file.resolve():
                    raise ValueError("Batch output must differ from input")
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                with args.output_file.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                    writer.writerow(("source", "translation", "alternatives", "method", "elapsed_ms", "warnings"))
                    for source in sources:
                        result = translator.translate(source, args.method, args.top_k, args.beam_size)
                        first = result.translations[0].text if result.translations else ""
                        writer.writerow((source, first, json.dumps([asdict(c) for c in result.translations],
                                        ensure_ascii=False), result.method, result.elapsed_ms, " | ".join(result.warnings)))
            else:
                for source in sources:
                    show(translator.translate(source, args.method, args.top_k, args.beam_size), args.json)
        else:
            interactive(translator, args)
    finally:
        translator.close()


def interactive(translator, args):
    timing = True
    while True:
        try:
            text = input("English> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text.startswith(":"):
            show(translator.translate(text, args.method, args.top_k, args.beam_size), args.json, timing)
            continue
        command, _, value = text.strip().partition(" ")
        value = value.strip()
        if command in (":q", ":quit"):
            return
        if command == ":help":
            print(":quit | :method auto|dictionary|nmt | :beam N | :topk N | :timing on|off | :clear | :config")
        elif command == ":method" and value in METHODS:
            args.method = value
        elif command in (":beam", ":topk") and value.isdecimal() and 1 <= int(value) <= 50:
            setattr(args, "beam_size" if command == ":beam" else "top_k", int(value))
        elif command == ":timing" and value in ("on", "off"):
            timing = value == "on"
        elif command == ":config":
            print(json.dumps(vars(args), default=str, indent=2))
        elif command == ":clear":
            print("\033[2J\033[H", end="")
        else:
            print("Invalid command. Enter :help for commands.")
