from collections import Counter, defaultdict
import json
from importlib.metadata import version
import os
import random
from statistics import mean, median, quantiles
from time import perf_counter

from sacrebleu.metrics import BLEU, CHRF

from .data import bucket, digest, read, report, write
from .inference import Translator, english_words
from .text import normalize


def scores(hypotheses, references):
    bleu = BLEU(tokenize="none")
    chrf = CHRF()
    chrf_plus = CHRF(word_order=2)
    hypotheses, references = list(map(normalize, hypotheses)), list(map(normalize, references))
    f1 = []
    for hypothesis, reference in zip(hypotheses, references):
        left, right = hypothesis.split(), reference.split()
        overlap = sum((Counter(left) & Counter(right)).values())
        f1.append(2 * overlap / max(1, len(left) + len(right)))
    return {"bleu": bleu.corpus_score(hypotheses, [references]).score,
            "chrf": chrf.corpus_score(hypotheses, [references]).score,
            "chrf++": chrf_plus.corpus_score(hypotheses, [references]).score,
            "exact_match": 100 * sum(a == b for a, b in zip(hypotheses, references)) / len(references),
            "token_f1": 100 * sum(f1) / len(references)}


def intervals(hypotheses, references, samples):
    rng = random.Random(42)
    metrics = {"bleu": BLEU(tokenize="none"), "chrf": CHRF()}
    hypotheses, references = list(map(normalize, hypotheses)), list(map(normalize, references))
    estimates = defaultdict(list)
    for _ in range(samples):
        indices = rng.choices(range(len(references)), k=len(references))
        h, r = [hypotheses[i] for i in indices], [references[i] for i in indices]
        for key, metric in metrics.items():
            estimates[key].append(metric.corpus_score(h, [r]).score)
    return {key: quantiles(values, n=40, method="inclusive")[::38] for key, values in estimates.items()}


def latency(values):
    return {"median_ms": median(values), "p95_ms": sorted(values)[max(0, (95 * len(values) + 99) // 100 - 1)]}


def peak_working_set():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_working_set", "working_set", "peak_paged", "paged",
             "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    get_memory = kernel.K32GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    counters = Counters()
    counters.size = ctypes.sizeof(counters)
    if get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.size):
        return counters.peak_working_set / 2**20
    return None


def calibrate(model, tokenizer, train, validation, beam):
    rows = read(validation)
    translator = Translator(None, model, tokenizer)
    neural = translator.load_neural()
    validation_hash = digest(validation)
    if validation_hash != neural.validation_hash:
        raise ValueError("Calibration requires the validation split recorded in the checkpoint")
    if digest(train) != neural.training_hash:
        raise ValueError("Calibration requires the training split recorded in the checkpoint")
    vocabulary = {word for _, source, _ in read(train) for word in english_words(source)}
    unknown_word_counts = [len(set(english_words(source)) - vocabulary) for _, source, _ in rows]
    metric = CHRF()
    ranked_rows, unfiltered = [], []
    for reference, source, _ in rows:
        ranked = neural.ranked(source, 3, beam)
        reference = normalize(reference)
        ranked_rows.append([(metric.sentence_score(normalize(candidate.text), [reference]).score, nll)
                            for candidate, nll in ranked])
        unfiltered.append(ranked[0][0].text if ranked else "")
    baseline = mean((ranked[0][0] if ranked else 0) for ranked in ranked_rows)
    policy = {"checkpoint_sha256": digest(model), "validation_sha256": validation_hash, "beam_size": beam,
              "validation_rows": len(rows), "unfiltered_validation_metrics": scores(unfiltered, [row[0] for row in rows]),
              "unfiltered_mean_sentence_chrf": baseline, "max_mean_token_nll": None, "max_unseen_words": None}
    # Pick the most coverage that still adds four sentence-chrF points on
    # validation while retaining at least three quarters of the requests.
    best = (0, float("-inf"))
    for vocabulary_limit in (0, 1, 2):
        for cutoff in (step / 20 for step in range(4, 41)):
            retained = [next((score for score, nll in row if nll <= cutoff), None)
                        for row, unknown in zip(ranked_rows, unknown_word_counts) if unknown <= vocabulary_limit]
            retained = [score for score in retained if score is not None]
            if len(retained) < .75 * len(rows):
                continue
            average = mean(retained)
            if average >= baseline + 4 and (len(retained), average) > best:
                best = (len(retained), average)
                policy.update(max_mean_token_nll=cutoff, max_unseen_words=vocabulary_limit,
                              validation_returned=len(retained), returned_mean_sentence_chrf=average)
    if policy["max_unseen_words"] is not None:
        policy["training_english_vocabulary"] = sorted(vocabulary)
    report(model.with_name("quality_policy.json"), [json.dumps(policy, indent=2)])
    print(json.dumps({k: v for k, v in policy.items() if k != "training_english_vocabulary"}, indent=2))


def evaluate(model, tokenizer, test, dictionary, production, output, beam, samples):
    from .dictionary import Dictionary
    if samples < 2:
        raise ValueError("At least two bootstrap samples are required")
    rows = read(test)
    if not rows:
        raise ValueError("Empty test split")
    started = perf_counter()
    translator = Translator(dictionary, model, tokenizer)
    production_db = Dictionary(production) if production and production.exists() else None
    try:
        neural = translator.load_neural()
        if translator.dictionary.source_hash() != neural.training_hash:
            raise ValueError("Evaluation dictionary must be built from the checkpoint's training split")
        if digest(test) in {neural.training_hash, neural.validation_hash}:
            raise ValueError("Test CSV must differ from the checkpoint's training and validation CSVs")
        startup = (perf_counter() - started) * 1000
        predictions, by_mode = [], defaultdict(list)
        dictionary_times, exact_times, neural_times = [], [], []
        scored_on_test = []
        covered, production_covered = 0, 0
        for reference, source, category in rows:
            try:
                start = perf_counter()
                ranked = neural.ranked(source, 3, beam)
                elapsed = (perf_counter() - start) * 1000
            except ValueError as error:
                raise ValueError(f"Cannot evaluate {source!r}: {error}") from error
            nmt = ranked[0][0] if ranked else None
            reliable = neural.reliable(ranked, source)
            scored_on_test.append((reliable[0].text, reference) if reliable else None)
            dictionary_result = translator.translate(source, "dictionary")
            hit = next(iter(dictionary_result.translations), None)
            covered += bool(hit)
            if production_db:
                start = perf_counter()
                found = production_db.lookup(source)
                if found:
                    production_covered += 1
                    exact_times.append((perf_counter() - start) * 1000)
            chosen = hit or nmt
            record = (reference, source, category)
            predictions.append((source, reference, chosen.text if chosen else "", chosen.source if chosen else "no_match",
                                chosen.score if chosen else None, elapsed if not hit else dictionary_result.elapsed_ms,
                                category, bucket(record)))
            for mode, candidate in (("nmt", nmt), ("dictionary", hit), ("hybrid", chosen)):
                by_mode[mode].append(candidate.text if candidate else "")
            neural_times.append(elapsed)
            dictionary_times.append(dictionary_result.elapsed_ms)
        references = [row[0] for row in rows]
        measurements = {mode: scores(texts, references) | {"bootstrap_95": intervals(texts, references, samples)}
                        for mode, texts in by_mode.items()}
        meta = {"test_rows": len(rows), "test_sha256": digest(test), "train_dictionary_coverage": covered / len(rows),
                "production_coverage_includes_test": production_covered / len(rows) if production_db else None,
                "sacrebleu": version("sacrebleu"), "bleu_tokenization": "normalized whitespace",
                "bootstrap_samples": samples, "bootstrap_seed": 42, "modes": measurements}
        if neural.max_token_nll is not None:
            accepted = [pair for pair in scored_on_test if pair]
            metric = CHRF()
            meta["accuracy_cutoff"] = {
                "max_mean_token_nll": neural.max_token_nll, "max_unseen_words": neural.max_unseen_words,
                "returned": len(accepted),
                "coverage": len(accepted) / len(rows),
                "all_mean_sentence_chrf": mean(metric.sentence_score(normalize(h), [normalize(r)]).score
                                              for h, r in zip(by_mode["nmt"], references)),
                "returned_mean_sentence_chrf": mean(metric.sentence_score(normalize(h), [normalize(r)]).score
                                                   for h, r in accepted) if accepted else None,
            }
        kinds = defaultdict(list)
        for i, row in enumerate(rows):
            kinds["category", row[2]].append(i)
            kinds["length", bucket(row)].append(i)
        segments = []
        for (kind, group), indices in sorted(kinds.items()):
            for mode, texts in by_mode.items():
                result = scores([texts[i] for i in indices], [references[i] for i in indices])
                segments.append((kind, group, mode, len(indices), *result.values()))
        write(output / "metrics_by_category.csv",
              ("group_type", "group", "mode", "rows", "bleu", "chrf", "chrf++", "exact_match", "token_f1"), segments)
        write(output / "test_predictions.csv",
              ("source", "reference", "hypothesis", "method", "score", "elapsed_ms", "category", "length_bucket"), predictions)
        report(output / "metrics.json", [json.dumps(meta, indent=2)])
        sweep = []
        from .model import threads
        for thread_count in (1, 4, 6, 8):
            threads(thread_count)
            for width in dict.fromkeys((1, beam)):
                times = []
                for _, source, _ in rows[:min(10, len(rows))]:
                    start = perf_counter()
                    neural.ranked(source, 1, width)
                    times.append((perf_counter() - start) * 1000)
                sweep.append(f"{thread_count} threads, beam {width}: {latency(times)}")
        peak = peak_working_set()
        report(output / "latency_report.txt",
               [f"Startup: {startup:.1f} ms", f"Checkpoint: {model.stat().st_size / 1e6:.2f} MB",
                f"Evaluation process peak working set: {f'{peak:.1f} MiB' if peak is not None else 'unavailable'}",
                f"NMT beam {beam}: {latency(neural_times)}",
                f"Train-only dictionary queries: {latency(dictionary_times)}",
                f"Production dictionary exact hits: {latency(exact_times) if exact_times else 'unavailable'}",
                "Thread sweep (first 10 test pairs)", *sweep])
        report(output / "error_analysis.txt",
               ["Single-reference test results may count valid alternatives as errors.",
                f"Empty NMT hypotheses: {sum(not text for text in by_mode['nmt'])}",
                "Sample disagreements (source | reference | prediction)",
                *(" | ".join(row[:3]) for row in predictions if normalize(row[1]) != normalize(row[2]))][:13])
    finally:
        translator.close()
        if production_db:
            production_db.close()
