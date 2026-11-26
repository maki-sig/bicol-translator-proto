import csv
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
import random
import re
import unicodedata

from .text import fold, lookup_key, normalize

FIELDS = ("Bikol-Naga", "English Translation", "Category")
WORDS = set("Noun Verb Adjective Number Time Direction Color Pronoun Grammar Expression".split())
TYPES = WORDS | set("Sentence Everyday Greeting Conversation Question Response Request Politeness Apology Farewell Introduction School Food Home Market Shopping Family Compliment Phrase".split())
SPLITS = ("train", "val", "test")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != FIELDS:
            raise ValueError(f"Invalid schema in {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No translation pairs in {path}")
    for line, row in enumerate(rows, 2):
        if len(row) != 3 or not all(field.strip() for field in row) or row[2] not in TYPES:
            raise ValueError(f"Invalid pair at {path}:{line}")
        if any(unicodedata.category(char).startswith("C") for field in row for char in field):
            raise ValueError(f"Control character at {path}:{line}")
        if any(re.search(r"Bikol\s*-\s*Naga\s*(?:Sentence\s*)?English\s*Translation\s*Category|Common Household Objects & Places", field, re.I) for field in row):
            raise ValueError(f"Header at {path}:{line}")
    return [tuple(row) for row in rows]


def write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def report(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, lines)) + "\n", encoding="utf-8", newline="\n")


def pair_key(row):
    return tuple(map(fold, row[:2]))


def bucket(row):
    size = max(len(text.split()) for text in row[:2])
    return "short" if size <= 4 else "medium" if size <= 8 else "long"


def groups(rows, threshold=.9):
    parents = list(range(len(rows)))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    # Also group exact normalized English inputs, including alternate references.
    english = {}
    for i, row in enumerate(rows):
        key = lookup_key(row[1])
        parents[i] = root(english.setdefault(key, i))
    texts = sorted(((" \n ".join(map(normalize, row[:2])), i) for i, row in enumerate(rows)),
                   key=lambda item: len(item[0]))
    matcher = SequenceMatcher(autojunk=False)
    for pos, (left, i) in enumerate(texts):
        matcher.set_seq2(left)
        for jpos in range(pos + 1, len(texts)):
            right, j = texts[jpos]
            if 2 * len(left) < threshold * (len(left) + len(right)):
                break
            if root(i) == root(j):
                continue
            matcher.set_seq1(right)
            if matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold:
                parents[root(j)] = root(i)
    result = defaultdict(list)
    for i in range(len(rows)):
        result[root(i)].append(i)
    return list(result.values())


def verify(full, sentences, review, output, verified=None, build_report=None):
    dictionary, corpus = read(full), read(sentences)
    dictionary_keys = set(map(pair_key, dictionary))
    corpus_keys = set(map(pair_key, corpus))
    if len(dictionary_keys) != len(dictionary) or len(corpus_keys) != len(corpus):
        raise ValueError("Duplicate translation pairs")
    if not set(corpus) <= set(dictionary):
        raise ValueError("Sentence corpus is not a subset of the full corpus")
    with review.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS) <= set(reader.fieldnames or ()):
            raise ValueError(f"Invalid review schema in {review}")
        rejected = {pair_key((row[FIELDS[0]], row[FIELDS[1]])) for row in reader
                    if row[FIELDS[0]] and row[FIELDS[1]]}
    if rejected & corpus_keys:
        raise ValueError("Reviewed pairs overlap with the training corpus")
    lines = [f"{path}  SHA-256 {digest(path)}" for path in (full, sentences, review)]
    if build_report and build_report.exists():
        lines.append(f"{build_report}  SHA-256 {digest(build_report)}")
    lines += [f"Full rows: {len(dictionary)}", f"Sentence rows: {len(corpus)}",
              f"Review pair overlaps: {len(rejected & corpus_keys)}",
              f"Categories: {dict(sorted(Counter(row[2] for row in corpus).items()))}"]
    for side in range(2):
        for label, length in (("tokens", lambda s: len(s.split())), ("characters", len)):
            values = [length(row[side]) for row in corpus]
            lines.append(f"{FIELDS[side]} {label}: min={min(values)} mean={sum(values)/len(values):.2f} max={max(values)}")
    lines.append("Warnings (shared markers such as mga and para can be valid Bikol)")
    for index, row in enumerate(dictionary, 2):
        marks = re.findall(r"\b(?:ang|ng|mga|para|pagkain|hindi|tulong)\b", fold(row[0]))
        if marks or fold(row[0]) == fold(row[1]):
            lines.append(f"Full CSV line {index}: {', '.join(sorted(set(marks))) or 'identical sides'} | {row[0]}")
    lines.append("Near-duplicate groups (sentence CSV lines)")
    lines.extend(", ".join(str(i + 2) for i in group) for group in groups(corpus) if len(group) > 1)
    report(output, lines)
    if verified:
        write(verified, FIELDS, corpus)


def stratum(row):
    category = row[2]
    if category in {"Conversation", "Response", "Request", "Politeness"}:
        category = "Dialogue"
    elif category in {"Everyday", "Sentence"}:
        category = "Sentence"
    elif category not in {"Greeting", "Question", "Phrase"}:
        category = "Other"
    return category, bucket(row)


def split(rows, seed, ratios, threshold):
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-6 or not 0 < threshold <= 1:
        raise ValueError("Split ratios must be positive and sum to 1, and similarity must be in (0, 1]")
    clusters = groups(rows, threshold)
    random.Random(seed).shuffle(clusters)
    clusters.sort(key=len, reverse=True)
    totals = Counter(map(stratum, rows))
    targets = [ratio * len(rows) for ratio in ratios]
    counts = [Counter() for _ in SPLITS]
    assignments = [[] for _ in SPLITS]
    for cluster in clusters:
        types = Counter(stratum(rows[i]) for i in cluster)
        def cost(k):
            remaining = targets[k] - len(assignments[k])
            local = sum(n * (ratios[k] * totals[key] - counts[k][key]) / totals[key]
                        for key, n in types.items()) / len(cluster)
            return remaining / max(targets[k], 1) + local
        dest = max(range(3), key=cost)
        assignments[dest].extend(cluster)
        counts[dest].update(types)
    return {name: sorted(indices) for name, indices in zip(SPLITS, assignments)}, clusters


def split_files(source, output, seed, ratios, threshold):
    rows = read(source)
    assignments, clusters = split(rows, seed, ratios, threshold)
    lines = [f"Seed: {seed}", f"Source SHA-256: {digest(source)}", f"Ratios: {ratios}",
             f"Similarity: {threshold}", f"Groups: {len(clusters)}"]
    membership = {}
    for name, indices in assignments.items():
        selected = [rows[i] for i in indices]
        write(output / f"{name}.csv", FIELDS, selected)
        membership.update((i, name) for i in indices)
        lines.append(f"{name}: {len(indices)} | categories {dict(sorted(Counter(row[2] for row in selected).items()))} | lengths {dict(Counter(map(bucket, selected)))}")
    if len(assignments["test"]) < 80:
        lines.append("Test split contains fewer than 80 pairs. A smaller test ratio would reduce coverage further.")
    write(output / "groups.csv", ("group_id", "source_line", "split"),
          ((group_id, i + 2, membership[i]) for group_id, cluster in enumerate(clusters, 1) for i in cluster))
    report(output / "split_report.txt", lines)


def preprocess(source, output):
    lines = []
    for name in SPLITS:
        rows = read(source / f"{name}.csv")
        for lang, side in (("en", 1), ("bk", 0)):
            texts = [normalize(row[side]) for row in rows]
            report(output / f"{name}.{lang}.txt", texts)
            lines.append(f"{name}.{lang}: {len(texts)} rows, {sum(map(len, texts))} characters, {sum(len(t.split()) for t in texts)} tokens")
    report(output / "preprocessing_report.txt", lines)
