"""Compare the local custom and MarianMT engines in separate CPU processes."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bikol_nmt.cli import MARIAN_DIR, MODEL, SPLITS, TOKENIZER, positive_int
from bikol_nmt.data import digest, read, report, write
from bikol_nmt.evaluation import latency, peak_working_set, scores
from bikol_nmt.inference import ENGINES, Translator
from bikol_nmt.marian import MODEL_ID


def cpu_name():
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def worker(args):
    rows = read(args.test)
    test_hash = digest(args.test)
    before = peak_working_set()
    translator = Translator(None, args.model, args.tokenizer, {"threads": args.threads},
                            engine=args.worker, marian_dir=args.marian_dir)
    try:
        start = perf_counter()
        neural = translator.load_neural()
        load_ms = (perf_counter() - start) * 1000
        after_load = peak_working_set()
        if args.worker == "custom" and test_hash in {neural.training_hash, neural.validation_hash}:
            raise ValueError("Choose a test split separate from the custom model's training and validation data")
        neural.ranked(rows[0][1], 1, args.beam_size)
        hypotheses, times = [], []
        for _, source, _ in rows:
            start = perf_counter()
            ranking = neural.ranked(source, 1, args.beam_size)
            times.append((perf_counter() - start) * 1000)
            hypotheses.append(ranking[0][0].text if ranking else "")
        peak = peak_working_set()
        metrics = scores(hypotheses, [row[0] for row in rows])
        return {
            "test_sha256": test_hash,
            "rows": len(rows),
            "weights_bytes": (args.marian_dir / "pytorch_model.bin" if args.worker == "marian"
                              else args.model).stat().st_size,
            "parameters": neural.parameter_count,
            "max_new_tokens": neural.settings["max_len"] - 1,
            "load_ms": load_ms, "latency": latency(times),
            "peak_before_load_mib": before, "peak_after_load_mib": after_load,
            "peak_working_set_mib": peak,
            "empty_outputs": sum(not result for result in hypotheses),
            "scores": metrics,
            "hypotheses": hypotheses, "row_latency_ms": times,
        }
    finally:
        translator.close()


def run_worker(args, engine):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", engine,
               "--test", str(args.test), "--model", str(args.model), "--tokenizer", str(args.tokenizer),
               "--marian-dir", str(args.marian_dir), "--threads", str(args.threads),
               "--beam-size", str(args.beam_size)]
    run = subprocess.run(command, capture_output=True, encoding="utf-8", check=False,
                         env=os.environ | {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                                           "PYTHONIOENCODING": "utf-8"})
    if run.returncode:
        raise RuntimeError(f"{engine} benchmark failed: {run.stderr.strip() or run.stdout.strip()}")
    return json.loads(run.stdout)


def compare(args):
    models = {}
    for engine in ENGINES:
        print(f"Benchmarking {engine} on CPU...", flush=True)
        models[engine] = run_worker(args, engine)
    rows = read(args.test)
    checksum = digest(args.test)
    if any(model["test_sha256"] != checksum or model["rows"] != len(rows) for model in models.values()):
        raise ValueError("The test file changed during the benchmark")
    predictions = []
    for i, (reference, source, category) in enumerate(rows):
        predictions.append((source, reference, category, models["custom"]["hypotheses"][i],
                            models["marian"]["hypotheses"][i], models["custom"]["row_latency_ms"][i],
                            models["marian"]["row_latency_ms"][i]))
    write(args.output_dir / "predictions.csv",
          ("source", "reference", "category", "custom", "marian", "custom_ms", "marian_ms"), predictions)
    for model in models.values():
        model.pop("hypotheses")
        model.pop("row_latency_ms")
    for engine, weights in (("custom", args.model), ("marian", args.marian_dir / "pytorch_model.bin")):
        checksum_weights = sha256()
        with weights.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                checksum_weights.update(block)
        models[engine]["weights_sha256"] = checksum_weights.hexdigest()
    test_path = args.test.resolve()
    metadata = {
        "test_file": test_path.relative_to(ROOT).as_posix() if test_path.is_relative_to(ROOT) else str(test_path),
        "test_rows": len(rows), "test_sha256": checksum,
        "platform": platform.platform(), "cpu": cpu_name(),
        "versions": {package: version(package) for package in ("torch", "transformers", "sacrebleu")},
        "marian_model_id": MODEL_ID,
        "threads": args.threads, "beam_size": args.beam_size, "warmup_rows_per_engine": 1,
        "method": "Unfiltered top NMT result, without dictionary or custom confidence rules",
        "memory_note": "Peak working set from a fresh process for each engine, including model loading",
        "pretraining_note": "The project's test split was held out from custom training. "
                            "Overlap with the upstream MarianMT training corpus is unverified.",
        "models": models,
    }
    report(args.output_dir / "metrics.json", [json.dumps(metadata, indent=2)])
    header = f"{'Engine':<10}{'BLEU':>8}{'chrF':>8}{'Median ms':>12}{'P95 ms':>10}{'Peak MiB':>12}"
    print(header)
    for engine, model in models.items():
        print(f"{engine:<10}{model['scores']['bleu']:>8.2f}{model['scores']['chrf']:>8.2f}"
              f"{model['latency']['median_ms']:>12.1f}{model['latency']['p95_ms']:>10.1f}"
              f"{model['peak_working_set_mib']:>12.1f}")
    print(f"Reports written to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / SPLITS / "test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / MODEL)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / TOKENIZER)
    parser.add_argument("--marian-dir", type=Path, default=ROOT / MARIAN_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/benchmark")
    parser.add_argument("--beam-size", type=int, choices=range(1, 51), default=5, metavar="N")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--worker", choices=ENGINES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            print(json.dumps(worker(args)))
        else:
            compare(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
