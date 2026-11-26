"""Build the offline Bikol-Naga/English corpus (Python 3.10+).

Run from the repository root:
    python scripts/build_nmt_csv.py --overrides data/manual_overrides.csv
    python scripts/build_nmt_csv.py --overrides data/manual_overrides.csv --sentences-output data/processed/bikol_naga_sentences.csv

--sentences-only filters the main output. --sentences-output writes a
separate filtered file, so it can be used alongside the full output.
--dry-run prints the report and leaves all files alone.

An override can include a source_line (the first line of match_text). Several
corrections for one damaged block may share the same match_text and source_line.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
from pathlib import Path
import re
import sys
import unicodedata


FIELDS = ("Bikol-Naga", "English Translation", "Category")
REVIEW_FIELDS = ("raw_text", *FIELDS, "confidence", "reason", "source_line")
SENTENCE_CATEGORIES = frozenset("""
    Sentence Everyday Greeting Conversation Question Response Request
    Politeness Apology Farewell Introduction School Food Home Market Shopping
    Family Compliment Phrase
""".split())
WORD_CATEGORIES = frozenset("""
    Noun Verb Adjective Number Time Direction Color Pronoun Grammar Expression
""".split())
CATEGORY_MAP = {name.casefold(): name for name in SENTENCE_CATEGORIES | WORD_CATEGORIES}
CATEGORY_MAP["adj"] = "Adjective"
HEADER_PATTERNS = re.compile(
    r"Bikol\s*-\s*Naga\s*(?:Sentence\s*)?English\s*Translation\s*Category"
    r"|Common Household Objects\s*&\s*Places|^[ \t]*Category[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
COMMON_CATEGORY_WORDS = {"Time", "School", "Food", "Home", "Market", "Family",
                         "Shopping", "Question", "Response", "Conversation",
                         "Sentence", "Phrase", "Direction", "Color", "Number",
                         "Expression", "Greeting", "Request", "Compliment"}
CATEGORY_ANCHOR = re.compile(
    r"Conversatio[ \t]*n|" + "|".join(
        sorted((name for name in CATEGORY_MAP.values() if name != "Conversation"),
               key=lambda name: (-len(name), name))
    ) + r"|Adj"
)
CATEGORY_FIELD = re.compile(r"[|;\t]| {2,}")
UNKNOWN_LABEL = re.compile(r"[A-Z][\w-]*(?: [A-Z][\w-]*){0,2}")
WHITESPACE = re.compile(r"\s*")
WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
SENTENCE_END = re.compile(r"""[.!?]+["”]?(?=\s|[A-ZÀ-ÖØ-Þ]|$)(?![ \t]*/|[ \t]*\()""")
CHARACTER_MAP = str.maketrans({"’": "'", "‘": "'", "–": "-", "—": "-", "\u00a0": " "})

BIKOL_MARKERS = frozenset("""
    nin asin nganing sagkod kan gabos hali dai dae iyo giraray gayo garo daw
    palan nindo ninda saimo saimong saiya saiyang sainda saindong saindo sako
    sakong satong sato simbag saod saudan laog ngunyan kasuudma harong paninda
    paratukdo banwaan paduman kaidto mabalos marhay kakanon eskwelahan aki
    ako ika ikaw kamo kita kami sinda niya anong baretan boot ho tabi po
""".split())
BIKOL_FUNCTION_WORDS = frozenset("""
    an sa si ni mga na pa man sana ta o para yang ka ko mo kun may kong
""".split())
BIKOL_STARTERS = (BIKOL_MARKERS | frozenset("""
    ano siisay sain saen nuarin kasuarin ta'no pano gurano pira dios salamat
    kumusta ingat paaram taga gusto aram igwa duman dini diyan ini iyan ito
    saro duwa tulo wala igdi isinurat nakatago pirang sa an
""".split()))
BIKOL_PREFIX = re.compile(r"(?:nag|mag|pag|pig|pina|maka|naka|naga|naki)[a-z-]{2}")

ENGLISH_MARKERS = frozenset("""
    the a i you he she it we they me him us them my your his her our their
    this that these those of to in on at for from with and or but because
    when where who what why how which is are was were be been have has had do
    does did will would can could should might must if as up out away before
    after through into across about very too more much many inside around
    until above below over toward under only always just during while already
    there here now please let's i'll i'm don't you're that's
""".split())
ENGLISH_STARTERS = ENGLISH_MARKERS | frozenset("""
    an let good thank thanks god no yes father mother grandmother grandfather
    everyone everything children take welcome goodbye write read use brush
    okay excuse happy pleased buy drink sweep put coffee eating walking working
    choose harvest speak study listen think come get rinse clean show
""".split())
ENGLISH_GLOSS_STARTERS = frozenset("""
    arm ash attitude bag back believe black blood blue body bright brown cooked
    cool day delicious down early empty fair family farm fast fine free front
    hand happy heavy high hope hot kind kitchen light long love low market
    mind near new night old one peace red right sad sea short slow soft sound
    strong sun sweet tall time tomorrow town very warm water well white work
    yesterday young joy little large small dark wind life month year late god
    land iron effort careful separate narrow broad open quiet thick wild money
    half full deep clear hard quick raw wrong cold return far
    advice appearance appreciation arrival beauty benefit book breakfast
    bravery care carefulness chicken child church clip community courage
    dream echo earthquake entertain exam expensive fellowship fiesta fish
    flower fragrance gathering garden hardness harvest heel helpful honor
    house human justice joyful kindness known lunch manage merchandise mount
    native office organize pathway pencil perseverance pleasant respect room
    scrub seashore service ship sorrow star stars story street table teacher
    togetherness trip uncooked understand vegetables weakness works yard
""".split())
ENGLISH_GLOSS_STARTERS |= ENGLISH_STARTERS
ENGLISH_VERBS = frozenset("""
    accompany adopt answer argue ask bathe be borrow buy change climb clean
    close converse cook dance disembark discuss drink drop eat fail fear fight
    fix fly follow forget fulfill gather get give go harvest heal help hope
    ignite improve join joke kiss lead learn lend listen live look lose make
    move obey open owe pass pay persevere pick plant play praise prepare
    quarrel read recover regret remember remain rent repair reside rest ride
    rinse run save say see sell separate share shout sing sit sleep slow smile
    speak spend stand stay stop study suffer sweep swim take talk teach tease
    think travel trust turn understand unite wait wake walk want wash waste
    watch win wipe work worry write
""".split())
TAGALOG_MARKERS = frozenset("""
    ang ng pagkain hindi tulong kapag ngunit sapagkat kanilang sinasabi
""".split())
ENGLISH_JOINS = {
    "similarto": "similar to", "howmuch": "how much", "howmany": "how many",
    "likethis": "like this", "turnoff": "turn off", "gohome": "go home",
    "gothere": "go there", "godown": "go down", "goup": "go up",
    "staythere": "stay there", "washhands": "wash hands",
    "washclothes": "wash clothes", "returninghome": "returning home",
    "borrowmoney": "borrow money", "lendmoney": "lend money",
    "slightlytired": "slightly tired", "slightlydirty": "slightly dirty",
    "slightlyshallow": "slightly shallow", "profoundlydeep": "profoundly deep",
    "fairlyeasy": "fairly easy", "fairlystraight": "fairly straight",
    "somewhathard": "somewhat hard", "veryfast": "very fast",
    "becareful": "be careful", "cookingpot": "cooking pot",
    "largepan": "large pan", "measuringbucket": "measuring bucket",
    "wildpig": "wild pig", "waterbuffalo": "water buffalo",
    "townsquare": "town square",
}


@dataclass(frozen=True)
class Text:
    value: str
    offset: int


@dataclass(frozen=True)
class Record:
    bikol: str
    english: str
    category: str
    source_line: int
    raw: str
    confidence: float = 1.0
    reason: str = ""
    manual: bool = False
    candidates: int = 1

    def fields(self) -> dict[str, str]:
        return dict(zip(FIELDS, (self.bikol, self.english, self.category)))


def normalize_layout(value: str) -> str:
    value = unicodedata.normalize("NFC", value.translate(CHARACTER_MAP))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        char for char in value
        if char in "\n\t" or unicodedata.category(char) not in {"Cc", "Cf"}
    )


def compact(value: str) -> str:
    return " ".join(value.split())


def normalize_text(value: str) -> str:
    value = compact(normalize_layout(value))
    value = re.sub(r"\s*/\s*", " / ", value)
    value = re.sub(r"(?<=\w)\(", " (", value)
    value = re.sub(r"\( +| +\)", lambda match: match[0].strip(), value)
    return value


def comparison_key(value: str) -> str:
    return normalize_text(value).casefold()


def normalize_english(value: str) -> str:
    value = normalize_text(value)
    match = re.match(r"To([a-z]+)", value)
    if match:
        tail = match[1]
        if (tail in ENGLISH_VERBS or tail in ENGLISH_JOINS
                or (value[match.end():].startswith(" / ") and tail not in {"morrow", "day", "night", "gether"})):
            value = f"To {tail}{value[match.end():]}"
    # A small set of unambiguous joins produced by this dump's column extraction.
    return WORD.sub(
        lambda match: (ENGLISH_JOINS[match[0].casefold()].capitalize()
                       if match[0][0].isupper() else ENGLISH_JOINS[match[0].casefold()])
        if match[0].casefold() in ENGLISH_JOINS else match[0], value,
    )


def canonical_category(raw: str, preserve: bool) -> str:
    name = "".join(raw.split()).casefold()
    if preserve and name == "adj":
        return "Adj"
    return CATEGORY_MAP.get(name, "")


def words(value: str) -> list[str]:
    return WORD.findall(value.casefold())


def language_signals(value: str) -> tuple[float, float]:
    terms = words(value)
    if not terms:
        return 0, 0
    bikol = sum(2 if word in BIKOL_MARKERS else 0.6 if word in BIKOL_FUNCTION_WORDS
                else 1.3 if BIKOL_PREFIX.match(word) else 0 for word in terms)
    english = sum(2 for word in terms if word in ENGLISH_MARKERS)
    if terms[0] in BIKOL_STARTERS or terms[0] in TAGALOG_MARKERS:
        bikol += 1.5
    if terms[0] in ENGLISH_STARTERS:
        english += 2
    elif terms[0] in ENGLISH_GLOSS_STARTERS or terms[0] in ENGLISH_VERBS:
        english += 1
    return bikol, english


def language(value: str) -> str:
    bikol, english = language_signals(value)
    if abs(bikol - english) < 1.4:
        return ""
    return "bikol" if bikol > english else "english"


def extract_category_anchors(value: str, preserve: bool):
    for match in CATEGORY_ANCHOR.finditer(value):
        start, end = match.span()
        following = WHITESPACE.match(value, end).end()
        next_char = value[following:following + 1]
        first = WORD.match(value, following)
        new_line = "\n" in value[end:following]
        category = canonical_category(match[0], preserve)
        if category in COMMON_CATEGORY_WORDS:
            # "School Noun" and "Family / Household Noun" are glosses.
            following_category = CATEGORY_ANCHOR.match(value, following)
            if next_char in {"/", "(", "'", "-"}:
                continue
            if (following_category and not new_line
                    and canonical_category(following_category[0], False) in WORD_CATEGORIES):
                continue
        if next_char and not next_char.isupper():
            if new_line and start and value[start - 1] in ".!?":
                yield start, end, category
                continue
            if (following == end or not first
                    or first[0] not in BIKOL_STARTERS
                    or first[0] in BIKOL_FUNCTION_WORDS):
                continue
        yield start, end, category


def parts_at(value: str, index: int) -> tuple[str, str]:
    return value[:index].rstrip(), value[index:].lstrip()


def split_score(left: str, right: str, glued: bool, category: str) -> float:
    left_words, right_words = words(left), words(right)
    if not left_words or not right_words:
        return float("-inf")
    if (left.endswith(("/", "-", "(", ";", "|")) or right.startswith(("/", "-", ")", "(", ";", "|"))
            or left.count("(") != left.count(")")):
        return float("-inf")
    left_bikol, left_english = language_signals(left)
    right_bikol, right_english = language_signals(right)
    first = right_words[0]
    score = 5 if right.lstrip('"“')[:1].isupper() else 0
    if glued:
        score += 1
    if left.endswith((".", "?", "!")):
        score += 6
    if first in ENGLISH_STARTERS:
        score += 2.5
    elif first in ENGLISH_GLOSS_STARTERS or first in ENGLISH_VERBS:
        score += 1.5
    if right.startswith("To") and category == "Verb":
        score += 3
    if first in BIKOL_MARKERS or (first in BIKOL_FUNCTION_WORDS - {"an"} and len(right_words) != 1):
        score -= 4
    score += min(left_bikol, 6) * .35 + min(right_english, 8) * .3
    score -= min(left_english, 10) * .9 + min(right_bikol, 10) * 1.25
    if left_words[-1] in BIKOL_FUNCTION_WORDS - {"man", "pa", "ka", "ko", "mo", "sana", "ta", "na"}:
        score -= 2
    if category in WORD_CATEGORIES and len(left_words) == 1:
        score += 1.2
    if len(left_words) == 1 and len(right_words) == 1:
        score += 1
    return score


def semicolon_boundary(left: str, right: str) -> bool:
    left_language, right_language = language(left), language(right)
    if left_language == right_language:
        return not left_language and len(words(left)) == len(words(right)) == 1
    if left_language != "bikol" and right_language != "bikol":
        return False
    bikol = left if left_language == "bikol" else right
    # A semicolon inside "Thank you; may God ..." belongs to the translation.
    return not any(
        match.start() and match[0][0].isupper() and match[0].casefold() in ENGLISH_STARTERS
        and language(bikol[match.start():]) == "english"
        for match in WORD.finditer(bikol)
    )


def explicit_pair(value: str) -> tuple[str, str] | None:
    for pattern in (r"\|", r"\t+", r" {2,}", r";"):
        fields = [part.strip() for part in re.split(pattern, value) if part.strip()]
        if len(fields) != 2:
            continue
        first, second = fields
        if pattern == ";" and not semicolon_boundary(first, second):
            continue
        if language(first) == "english" and language(second) == "bikol":
            first, second = second, first
        return first, second
    return None


def explicit_rows(value: str, preserve: bool) -> list[tuple[str, str, str]]:
    for pattern in (r"\|", r"\t+", r" {2,}", r";"):
        fields = [field.strip() for field in re.split(pattern, value.strip())]
        if len(fields) < 3 or len(fields) % 3 or not all(fields):
            continue
        triples = list(zip(fields[::3], fields[1::3], fields[2::3]))
        if not all(canonical_category(label, preserve) or UNKNOWN_LABEL.fullmatch(label)
                   for _, _, label in triples):
            continue
        if pattern == ";" and not all(semicolon_boundary(left, right) for left, right, _ in triples):
            continue
        return triples
    return []


def split_payload(value: str, category: str) -> tuple[str, str, float]:
    explicit = explicit_pair(value)
    if explicit:
        return *explicit, .99
    value = compact(value)
    boundaries = {match.end() for match in re.finditer(r"\s+", value)}
    boundaries.update(i for i in range(1, len(value)) if value[i-1].islower() and value[i].isupper())
    if re.fullmatch(r"[A-Z]{2}[a-z]+", value):
        boundaries.add(1)
    if not boundaries and category in WORD_CATEGORIES:
        # Weaker evidence for completely lowercase concatenations.
        for i in range(3, len(value) - 2):
            tail = value[i:].casefold()
            if tail in ENGLISH_GLOSS_STARTERS or tail in ENGLISH_VERBS or (
                category == "Verb" and tail.startswith("to") and tail[2:] in ENGLISH_VERBS
            ):
                boundaries.add(i)
    scored = []
    for index in sorted(boundaries):
        left, right = parts_at(value, index)
        glued = not value[index - 1].isspace()
        score = split_score(left, right, glued and right[:1].isupper(), category)
        scored.append((score, left, right))
        if language(left) == "english" and language(right) == "bikol":
            scored.append((split_score(right, left, False, category), right, left))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 1:
        return "", "", 0
    best, bikol, english = scored[0]
    second = scored[1][0] if len(scored) > 1 else best - 4
    margin = min(max(best - second, 0), 4)
    confidence = min(.98, .25 + min(best, 10) * .045 + margin * .10)
    return bikol, english, confidence


def segment_by_language(fragment: Text) -> list[tuple[str, Text]]:
    value = fragment.value
    pieces = []
    start = 0
    for match in SENTENCE_END.finditer(value):
        end = match.end()
        # Keep an opening question with its follow-up if both appear on one
        # physical line and the short question is clearly in the same language.
        if match[0] == "?" and "\n" not in value[start:end].strip() and len(words(value[start:end])) <= 4:
            next_sentence = re.match(r"[ \t]+([^\n.!?]+)[.!?]", value[end:])
            current_language = language(value[start:end])
            if next_sentence and current_language and language(next_sentence[1]) == current_language:
                continue
        pieces.append((start, end))
        start = end
    pieces.append((start, len(value)))
    units = []
    for start, end in pieces:
        piece = value[start:end]
        offset = fragment.offset + start
        unit = ""
        unit_offset = offset
        for line in piece.splitlines(keepends=True):
            part = line.strip()
            if part:
                kind = language(part)
                previous = language(unit)
                if unit and kind and previous and kind != previous and part[0].isupper():
                    units.append((previous, Text(unit, unit_offset)))
                    unit = ""
                if not unit:
                    unit_offset = offset + len(line) - len(line.lstrip())
                unit = f"{unit} {part}".strip()
            offset += len(line)
        if unit:
            units.append((language(unit), Text(unit, unit_offset)))
    return units


def raw_row(bikol: Text, english: Text, category: str, line: int, raw: str,
            confidence: float) -> Record:
    return Record(normalize_text(bikol.value), normalize_english(english.value),
                  category, line, raw, confidence)


class Parser:
    def __init__(self, source: str, preserve: bool):
        self.source = source
        self.preserve = preserve
        self.newlines = [match.end() for match in re.finditer("\n", source)]
        self.records: list[Record] = []
        self.bikol: list[Text] = []
        self.english: list[Text] = []
        self.categories: list[str] = []
        self.pending_start: int | None = None
        self.pending_end = 0

    def line(self, offset: int) -> int:
        return bisect_right(self.newlines, offset) + 1

    def reject_pending(self):
        if self.pending_start is None:
            return
        raw = self.source[self.pending_start:self.pending_end].strip()
        reason = (f"unaligned sentence block (Bikol: {len(self.bikol)}, "
                  f"English: {len(self.english)}, category labels: {len(self.categories)})")
        self.records.append(Record(
            normalize_text(" ".join(item.value for item in self.bikol)),
            normalize_text(" ".join(item.value for item in self.english)),
            ", ".join(self.categories), self.line(self.pending_start), raw, 0, reason,
            candidates=max(len(self.bikol), len(self.english), len(self.categories)),
        ))
        self.reset_pending()

    def reset_pending(self):
        self.bikol.clear()
        self.english.clear()
        self.categories.clear()
        self.pending_start = None

    def pair_sentences(self, units: list[tuple[str, Text]], category: str,
                       start: int, end: int):
        if self.pending_start is not None and units and units[0][0] == "bikol":
            # A new Bikol/English group after translations have already begun
            # means the old group has missing or interwoven text.
            if self.english:
                self.reject_pending()
        if self.pending_start is None:
            self.pending_start = start
        self.pending_end = end
        self.categories.append(category)
        for kind, part in units:
            (self.bikol if kind == "bikol" else self.english).append(part)
        if len(self.bikol) == len(self.english) == len(self.categories):
            raw = self.source[self.pending_start:end].strip()
            for bikol, english, label in zip(self.bikol, self.english, self.categories):
                line = self.line(min(bikol.offset, english.offset))
                self.records.append(raw_row(bikol, english, label, line, raw, .97))
            self.reset_pending()

    def consume(self, start: int, anchor: int, end: int, category: str):
        payload = self.source[start:anchor]
        start += len(payload) - len(payload.lstrip())
        payload = payload.strip()
        if category in SENTENCE_CATEGORIES and (payload or self.pending_start is not None):
            units = segment_by_language(Text(payload, start))
            if (all(kind for kind, _ in units) and not explicit_pair(payload)
                    and (self.pending_start is not None or len(units) >= 2
                         or (len(units) == 1 and len(words(payload)) >= 4
                             and payload.endswith((".", "?", "!"))))):
                if len(units) != 1 or self.pending_start is not None or "\n" in payload:
                    self.pair_sentences(units, category, start, end)
                    return
            if len(units) > 2:
                self.reject_pending()
                self.records.append(Record("", "", category, self.line(start),
                                           self.source[start:end].strip(), 0,
                                           "uncertain language in sentence block"))
                return
        self.reject_pending()
        raw = self.source[start:end].strip()
        bikol, english, confidence = split_payload(payload, category)
        self.records.append(raw_row(Text(bikol, start), Text(english, start), category,
                                    self.line(start), raw, confidence))

    def unknown(self, start: int, end: int):
        raw = self.source[start:end]
        offset = start + len(raw) - len(raw.lstrip())
        if not raw.strip():
            return
        fields = [field.strip() for field in CATEGORY_FIELD.split(raw.strip())]
        bikol, english, category = fields if len(fields) == 3 else ("", "", "")
        reason = "unknown or missing category"
        self.records.append(Record(normalize_text(bikol), normalize_english(english),
                                   category, self.line(offset), raw.strip(), 0, reason))

    def explicit_boundaries(self, occupied: list[tuple[int, int, list[Record]]]):
        offset = 0
        for raw_line in self.source.splitlines(keepends=True):
            end = offset + len(raw_line)
            line = raw_line.strip()
            rows = explicit_rows(line, self.preserve)
            if not rows:
                # An otherwise complete row with an unfamiliar final label
                # must not be absorbed into the next row's payload.
                label = re.search(r"[ \t]+([A-Z][\w-]+)$", line)
                if (label and not canonical_category(label[1], self.preserve)
                        and language(line) != "english"
                        and not any(extract_category_anchors(line, self.preserve))):
                    bikol, english, confidence = split_payload(line[:label.start()], "")
                    if confidence >= .65 and english[:1].isupper():
                        rows = [(bikol, english, label[1])]
            if rows and not any(start < end and stop > offset for start, stop, _ in occupied):
                replacements = []
                for left, right, label in rows:
                    if language(left) == "english" and language(right) == "bikol":
                        left, right = right, left
                    replacements.append(Record(
                        normalize_text(left), normalize_english(right),
                        canonical_category(label, self.preserve) or label,
                        self.line(offset), line, .99,
                    ))
                yield offset, end, replacements
            offset = end

    def parse(self, overrides: list[tuple[int, int, list[Record]]]) -> list[Record]:
        boundaries = [(match.start(), match.end(), []) for match in HEADER_PATTERNS.finditer(self.source)]
        boundaries.extend(overrides)
        boundaries.extend(self.explicit_boundaries(boundaries.copy()))
        position = 0
        for start, end, replacements in sorted(boundaries, key=lambda boundary: boundary[0]) + [
            (len(self.source), len(self.source), []),
        ]:
            if start < position:
                continue
            fragment_start = position
            for anchor, stop, category in extract_category_anchors(
                self.source[position:start], self.preserve,
            ):
                self.consume(fragment_start, position + anchor, position + stop, category)
                fragment_start = position + stop
            self.reject_pending()
            self.unknown(fragment_start, start)
            self.records.extend(replacements)
            position = end
        return self.records


def load_overrides(path: Path | None, parser: Parser) -> tuple[
    list[tuple[int, int, list[Record]]], list[str],
]:
    if path is None:
        return [], []
    warnings = []
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(("match_text", *FIELDS)).issubset(reader.fieldnames or ()):
            raise ValueError("overrides must have match_text and the three corpus columns")
        for row in reader:
            match_text = compact(normalize_layout(row.get("match_text") or ""))
            source_line = (row.get("source_line") or "").strip()
            if not match_text or (source_line and (not source_line.isdecimal() or int(source_line) < 1)):
                warnings.append(f"Override CSV line {reader.line_num}: invalid match_text or source_line")
                continue
            groups[match_text, int(source_line) if source_line else None].append(row)
    replacements = []
    for (match_text, source_line), rows in groups.items():
        pattern = r"\s+".join(map(re.escape, match_text.split()))
        matches = [
            match for match in re.finditer(pattern, parser.source)
            if source_line is None or parser.line(match.start()) == source_line
        ]
        if len(matches) != 1:
            warnings.append(f"Override for {match_text[:90]!r}: found {len(matches)} matches; expected one")
            continue
        start, end = matches[0].span()
        if any(start < other_end and other_start < end for other_start, other_end, _ in replacements):
            warnings.append(f"Override at source line {parser.line(start)} overlaps an earlier override")
            continue
        resolved = []
        for row in rows:
            category = (row.get("Category") or "").strip()
            resolved.append(Record(
                normalize_text(row.get("Bikol-Naga") or ""),
                normalize_english(row.get("English Translation") or ""),
                canonical_category(category, parser.preserve) or category,
                parser.line(start), matches[0][0], manual=True,
            ))
        replacements.append((start, end, resolved))
    return replacements, warnings


def validate_record(record: Record, min_confidence: float, max_length: int,
                    preserve: bool) -> tuple[str, ...]:
    if record.reason:
        return (record.reason,)
    bikol, english = record.bikol, record.english
    reasons = []
    if not canonical_category(record.category, preserve):
        reasons.append("unknown category")
    if not WORD.search(bikol) or not WORD.search(english):
        reasons.append("missing Bikol/English text or uncertain split")
    else:
        if comparison_key(bikol) == comparison_key(english):
            reasons.append("identical sides")
        if max(len(bikol), len(english)) > max_length:
            reasons.append(f"side exceeds {max_length} characters")
        if max(len(bikol), len(english)) > 5 * min(len(bikol), len(english)):
            reasons.append("length ratio exceeds 5:1")
        if not record.manual and len(words(bikol)) >= 2 and language(bikol) == "english":
            reasons.append("Bikol side looks English")
        if not record.manual and language(english) == "bikol":
            reasons.append("English side looks Bikol")
    if any(HEADER_PATTERNS.search(side) for side in (bikol, english)):
        reasons.append("header or section label")
    markers = sorted(set(words(bikol)) & TAGALOG_MARKERS)
    if markers:
        reasons.append("possible Tagalog: " + ", ".join(markers))
    if not record.manual and record.confidence < min_confidence and bikol and english:
        reasons.append("low split confidence")
    return tuple(reasons)


def deduplicate_records(records: list[Record]) -> tuple[list[Record], Counter, int]:
    unique = {}
    duplicates = Counter()
    conflicts = 0
    for record in records:
        key = comparison_key(record.bikol), comparison_key(record.english)
        if key in unique:
            duplicates[key] += 1
            conflicts += unique[key].category != record.category
        else:
            unique[key] = record
    return list(unique.values()), duplicates, conflicts


def sentence_rows(records: list[Record]) -> list[Record]:
    return [row for row in records if row.category in SENTENCE_CATEGORIES]


def write_csv(path: Path, fields: tuple[str, ...], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_report(source: Path, data: bytes, extracted: list[Record], accepted: list[Record],
                review: list[tuple[Record, tuple[str, ...]]], unique: list[Record],
                duplicates: Counter, conflicts: int, warnings: list[str], main_count: int,
                separate_count: int | None) -> str:
    candidate_count = sum(row.candidates for row in extracted)
    review_count = sum(row.candidates for row, _ in review)
    automatic_count = sum(not row.manual for row in accepted)
    automatic_rate = automatic_count / candidate_count if candidate_count else 0
    lines = [
        "Bikol-Naga NMT build report",
        f"Input: {source}",
        f"Input bytes: {len(data)}",
        f"Input SHA-256: {hashlib.sha256(data).hexdigest()}",
        f"Candidate records: {candidate_count}",
        f"Accepted before deduplication: {len(accepted)}",
        f"Accepted automatically: {automatic_count} ({automatic_rate:.1%})",
        f"Review rows: {len(review)} ({review_count} candidate records)",
        f"Duplicates removed: {sum(duplicates.values())}",
        f"Category conflicts: {conflicts}",
        f"Manual overrides applied: {sum(row.manual for row in extracted)}",
        f"Unique accepted pairs: {len(unique)}",
        f"Final output row count: {main_count}",
    ]
    if separate_count is not None:
        lines.append(f"Separate sentence output row count: {separate_count}")
    counts = Counter(row.category for row in unique)
    lines.extend(["", "Categories in the full corpus"])
    lines.extend(f"  {category}: {counts[category]}" for category in sorted(counts))
    lines.extend(["", "Review reasons and source lines"])
    reasons = Counter(reason for _, row_reasons in review for reason in row_reasons)
    lines.extend(f"  {count} x {reason}" for reason, count in sorted(reasons.items()))
    lines.extend(f"  Line {row.source_line}: {', '.join(row_reasons)} | {compact(row.raw)[:180]}"
                 for row, row_reasons in review)
    lines.extend(["", f"Top duplicate pairs ({len(duplicates)} groups)"])
    lines.extend(f"  {count + 1} copies | {left} | {right}"
                 for (left, right), count in duplicates.most_common(20))
    pairs = [(comparison_key(row.bikol), comparison_key(row.english), row.source_line) for row in unique]
    for side, label in enumerate(("Bikol with multiple English translations",
                                  "English with multiple Bikol translations")):
        alternatives = defaultdict(list)
        for pair in pairs:
            alternatives[pair[side]].append(pair[1 - side])
        groups = {key: group for key, group in alternatives.items() if len(group) > 1}
        lines.extend(["", f"{label} ({len(groups)} groups)"])
        lines.extend(f"  {key} | {'; '.join(group)}" for key, group in list(groups.items())[:20])
    near = near_duplicates(pairs)
    lines.extend(["", f"Near-duplicate pairs at similarity >= 0.90 ({len(near)} pairs)"])
    lines.extend(f"  {score:.3f} | Lines {first[2]} and {second[2]} | "
                 f"{first[0]} = {first[1]} | {second[0]} = {second[1]}"
                 for score, first, second in near)
    if warnings:
        lines.extend(["", "Warnings", *(f"  {warning}" for warning in warnings)])
    return "\n".join(lines) + "\n"


def near_duplicates(pairs: list[tuple[str, str, int]]) -> list[tuple]:
    # Sort by length so obviously dissimilar pairs never reach SequenceMatcher.
    items = sorted(((f"{left}\n{right}", (left, right, line)) for left, right, line in pairs),
                   key=lambda item: len(item[0]))
    matcher = SequenceMatcher(autojunk=False)
    matches = []
    for i, (first, first_pair) in enumerate(items):
        matcher.set_seq2(first)
        for j in range(i + 1, len(items)):
            second, second_pair = items[j]
            if 2 * len(first) < .90 * (len(first) + len(second)):
                break
            matcher.set_seq1(second)
            if matcher.quick_ratio() >= .90 and (similarity := matcher.ratio()) >= .90:
                matches.append((similarity, first_pair, second_pair))
    return sorted(matches, key=lambda item: (-item[0], item[1][2], item[2][2]))


def check_paths(inputs: list[Path], outputs: list[Path]):
    for i, output in enumerate(outputs):
        if output.is_dir():
            raise ValueError(f"output path is a directory: {output}")
        for other in inputs + outputs[:i]:
            if output.resolve() == other.resolve() or (
                output.exists() and other.exists() and output.samefile(other)
            ):
                raise ValueError(f"output path aliases another file: {output}")


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--input", type=Path, default=Path("data/raw_dataset.txt"))
    cli.add_argument("--output", type=Path, default=Path("data/processed/bikol_naga_nmt.csv"))
    cli.add_argument("--review", type=Path, default=Path("data/processed/bikol_naga_review.csv"))
    cli.add_argument("--report", type=Path, default=Path("data/processed/bikol_naga_report.txt"))
    cli.add_argument("--sentences-only", action="store_true", help="filter the main output to sentence categories")
    cli.add_argument("--sentences-output", type=Path, help="write a separate sentence-only CSV")
    cli.add_argument("--overrides", type=Path, help="apply the optional manual correction CSV")
    cli.add_argument("--min-confidence", type=float, default=.55)
    cli.add_argument("--max-length", type=int, default=300)
    cli.add_argument("--preserve-categories", action="store_true", help="keep Adj as the adjective label")
    cli.add_argument("--strict", action="store_true", help="exit 2 with review rows, or 3 above 5%%")
    cli.add_argument("--dry-run", action="store_true", help="print the report without writing files")
    cli.add_argument("--quiet", action="store_true", help="suppress the console summary")
    args = cli.parse_args(argv)
    if not 0 <= args.min_confidence <= 1 or args.max_length < 1:
        cli.error("min-confidence must be between 0 and 1, and max-length must be positive")
    try:
        inputs = [args.input] + ([args.overrides] if args.overrides else [])
        outputs = [args.output, args.review, args.report]
        if args.sentences_output:
            outputs.append(args.sentences_output)
        if not args.dry_run:
            check_paths(inputs, outputs)
        data = args.input.read_bytes()
        parser = Parser(normalize_layout(data.decode("utf-8-sig")), args.preserve_categories)
        overrides, warnings = load_overrides(args.overrides, parser)
        extracted = parser.parse(overrides)
        accepted = []
        review = []
        for row in extracted:
            reasons = validate_record(row, args.min_confidence, args.max_length, args.preserve_categories)
            if reasons:
                review.append((row, reasons))
            else:
                accepted.append(row)
        unique, duplicates, conflicts = deduplicate_records(accepted)
        sentences = sentence_rows(unique)
        final = sentences if args.sentences_only else unique
        report = make_report(args.input, data, extracted, accepted, review, unique, duplicates,
                             conflicts, warnings, len(final),
                             len(sentences) if args.sentences_output else None)
        if args.dry_run:
            if not args.quiet:
                print(report, end="")
        else:
            write_csv(args.output, FIELDS, (row.fields() for row in final))
            write_csv(args.review, REVIEW_FIELDS, (
                {"raw_text": compact(row.raw), **row.fields(), "confidence": f"{row.confidence:.3f}",
                 "reason": "; ".join(reasons), "source_line": row.source_line}
                for row, reasons in review
            ))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(report, encoding="utf-8", newline="\n")
            if args.sentences_output:
                write_csv(args.sentences_output, FIELDS, (row.fields() for row in sentences))
            if not args.quiet:
                print(f"Wrote {len(final)} unique pairs and {len(review)} review rows. Report: {args.report}")
                for warning in warnings:
                    print(warning, file=sys.stderr)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Build failed: {error}", file=sys.stderr)
        return 1
    if args.strict and review:
        return 3 if sum(row.candidates for row, _ in review) > len(accepted) / 19 else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
