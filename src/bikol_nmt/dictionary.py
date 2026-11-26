from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

from .data import WORDS, digest, read, report
from .text import lookup_key


def build(source, output):
    if source.resolve() == output.resolve() or output.exists() and output.samefile(source):
        raise ValueError("Dictionary output must differ from the source CSV")
    rows = read(source)
    entries = {}
    for bikol, english, category in rows:
        key = lookup_key(english), lookup_key(bikol)
        if not all(key):
            continue
        kind = "word" if category in WORDS or category == "Phrase" and len(english.split()) == 1 else (
            "phrase" if category == "Phrase" else "sentence")
        priority = ("word", "phrase", "sentence").index(kind) + 1
        old = entries.get(key)
        count = old[6] + 1 if old else 1
        if old and old[7] >= priority:
            entries[key] = (*old[:6], count, old[7])
        else:
            entries[key] = (english, bikol, category, kind, *key, count, priority)
    alternatives = Counter(key[0] for key in entries)
    info = {
        "schema_version": 1, "source_sha256": digest(source),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(entries), "duplicates": sum(row[6] - 1 for row in entries.values()),
        "alternative_keys": sum(n > 1 for n in alternatives.values()),
        "entry_types": dict(Counter(row[3] for row in entries.values())),
        "categories": dict(Counter(row[2] for row in entries.values())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage the file alongside its destination so it inherits the destination
    # directory's permissions on Windows when it is moved into place.
    with NamedTemporaryFile(dir=output.parent, suffix=".sqlite", delete=False) as staging:
        temporary = Path(staging.name)
    try:
        with closing(sqlite3.connect(temporary)) as connection, connection:
            connection.executescript("""
                CREATE TABLE entries (
                  id INTEGER PRIMARY KEY, english TEXT NOT NULL, bikol TEXT NOT NULL,
                  category TEXT NOT NULL, entry_type TEXT NOT NULL CHECK (entry_type IN ('word','phrase','sentence')),
                  normalized_english TEXT NOT NULL, normalized_bikol TEXT NOT NULL,
                  frequency INTEGER NOT NULL DEFAULT 1, priority INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX idx_entries_normalized_english ON entries(normalized_english);
                CREATE INDEX idx_entries_normalized_bikol ON entries(normalized_bikol);
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            connection.executemany("INSERT INTO entries VALUES (NULL,?,?,?,?,?,?,?,?)", entries.values())
            connection.executemany("INSERT INTO metadata VALUES (?,?)",
                                   ((key, json.dumps(value, sort_keys=True)) for key, value in info.items()))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    reports = output.with_name(output.stem + "_report.txt")
    report(reports, [json.dumps(info, indent=2, sort_keys=True)])
    report(output.parent / "dictionary_report.txt",
           [path.read_text(encoding="utf-8").rstrip() for path in sorted(output.parent.glob("*_report.txt"))
            if path.name != "dictionary_report.txt"])


class Dictionary:
    def __init__(self, path):
        path = path.resolve()
        try:
            self.connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        except sqlite3.OperationalError as error:
            raise OSError(f"Cannot open dictionary at {path}: {error}") from error
        self.query = lru_cache(maxsize=512)(self.uncached)

    def close(self):
        self.query.cache_clear()
        self.connection.close()

    def lookup(self, text):
        return self.query(lookup_key(text))

    def source_hash(self):
        result = self.connection.execute(
            "SELECT value FROM metadata WHERE key='source_sha256'").fetchone()
        return json.loads(result[0]) if result else None

    def uncached(self, key):
        return tuple(self.connection.execute(
            "SELECT bikol, category FROM entries WHERE normalized_english=? "
            "ORDER BY priority DESC, frequency DESC, id", (key,)))
