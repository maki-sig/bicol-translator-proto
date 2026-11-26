import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from scripts import build_nmt_csv as nmt


class ParserTests(unittest.TestCase):
    def parse(self, source, preserve=False):
        parser = nmt.Parser(nmt.normalize_layout(source), preserve)
        accepted, review = [], []
        for row in parser.parse([]):
            reasons = nmt.validate_record(row, .55, 300, preserve)
            if reasons:
                review.append((row, reasons))
            else:
                accepted.append((row.bikol, row.english, row.category))
        return accepted, review

    def test_inline_triples_headers_and_unicode(self):
        source = (
            "Bikol-Naga English Translation Category Ako I Pronoun Babae Woman Noun\n"
            "Bikol-NagaEnglishTranslationCategory LibroBookNoun KutingKitten/CatNoun "
            "MagkaonToeatVerb Mag-adalTostudy/LearnVerb BagaParticle(emphasis)Grammar "
            "Bag’oNewAdj Pin\u0303aPineappleNoun Sa agap Tomorrow Time\n"
            "Common Household Objects & Places\n"
            "Category\n"
        )
        accepted, review = self.parse(source)
        self.assertFalse(review)
        self.assertEqual(accepted, [
            ("Ako", "I", "Pronoun"), ("Babae", "Woman", "Noun"),
            ("Libro", "Book", "Noun"), ("Kuting", "Kitten / Cat", "Noun"),
            ("Magkaon", "To eat", "Verb"), ("Mag-adal", "To study / Learn", "Verb"),
            ("Baga", "Particle (emphasis)", "Grammar"), ("Bag'o", "New", "Adjective"),
            ("Piña", "Pineapple", "Noun"), ("Sa agap", "Tomorrow", "Time"),
        ])
        self.assertEqual(self.parse(source, preserve=True)[0][7][2], "Adj")

    def test_delimited_and_reversed_rows(self):
        accepted, review = self.parse(
            "Mag-ipit\n\tTo clip / hold together\n\tVerb\n"
            "\tMaghukad\n\tTo serve food from a pot\n\tVerb\n"
            "Tubig\tWater\tNoun\tAldaw\tDay\tNoun\n"
            "Kamot  Hand  Noun\n"
            "Harong | House | Noun\n"
            "Babae; Woman Noun\n"
            "I am tired. Pagal na ako. Everyday\n"
            "Dios mabalos po. Thank you; may God repay your kindness. Politeness\n"
        )
        self.assertFalse(review)
        self.assertEqual(accepted, [
            ("Mag-ipit", "To clip / hold together", "Verb"),
            ("Maghukad", "To serve food from a pot", "Verb"),
            ("Tubig", "Water", "Noun"), ("Aldaw", "Day", "Noun"),
            ("Kamot", "Hand", "Noun"), ("Harong", "House", "Noun"),
            ("Babae", "Woman", "Noun"), ("Pagal na ako.", "I am tired.", "Everyday"),
            ("Dios mabalos po.", "Thank you; may God repay your kindness.", "Politeness"),
        ])

    def test_wraps_column_order_and_false_category_words(self):
        accepted, review = self.parse(
            "Nag-aadal kami nin Bikol-Naga sa\n"
            "eskwelahan.\n"
            "Nagluto si nanay nin kakanon.\n"
            "We are studying Bikol-Naga in school.Everyday\n"
            "Mother cooked food. Everyday\n"
            "Mabalos saimo.Thank you.\n"
            "Conversatio n\n"
            "Masayang an panahon Time is wasted Phrase "
            "Eksamen sa eskwelahan Exam in school School "
            "Eskwelahan sa Naga School in Naga School "
            "Kakanon sa lamesa Food on the table Food "
            "Eskwelahan School Noun\n"
        )
        self.assertFalse(review)
        self.assertEqual(accepted, [
            ("Nag-aadal kami nin Bikol-Naga sa eskwelahan.",
             "We are studying Bikol-Naga in school.", "Everyday"),
            ("Nagluto si nanay nin kakanon.", "Mother cooked food.", "Everyday"),
            ("Mabalos saimo.", "Thank you.", "Conversation"),
            ("Masayang an panahon", "Time is wasted", "Phrase"),
            ("Eksamen sa eskwelahan", "Exam in school", "School"),
            ("Eskwelahan sa Naga", "School in Naga", "School"),
            ("Kakanon sa lamesa", "Food on the table", "Food"),
            ("Eskwelahan", "School", "Noun"),
        ])

    def test_review_isolates_ambiguous_blocks_and_unknown_categories(self):
        accepted, review = self.parse(
            "Nag-aadal kami sa eskwelahan.\n"
            "Naglakaw kami sa saod.\n"
            "We are studying at school. Everyday\n"
            "Tubig | Water | Unknown\n"
            "Babae Woman Unmapped\n"
            "Naglakaw ang mga aki. The children walked. Everyday\n"
            "Harong House Noun\n"
        )
        self.assertEqual(accepted, [("Harong", "House", "Noun")])
        self.assertEqual([row.source_line for row, _ in review], [1, 4, 5, 6])
        self.assertIn("unaligned sentence block", review[0][1][0])
        self.assertIn("unknown category", review[1][1])
        self.assertIn("unknown category", review[2][1])
        self.assertIn("possible Tagalog: ang", review[3][1])


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw.txt"
        self.full = self.root / "full.csv"
        self.review = self.root / "review.csv"
        self.report = self.root / "report.txt"
        self.sentences = self.root / "sentences.csv"
        self.args = [
            "--input", str(self.source), "--output", str(self.full),
            "--review", str(self.review), "--report", str(self.report),
        ]

    def write_source(self, text):
        self.source.write_bytes(text.encode("utf-8-sig"))

    def rows(self, path):
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.reader(handle))

    def test_csv_and_report_are_deterministic(self):
        self.write_source(
            'Ako I Pronoun Ako I Noun\r\n'
            'Ano an sabi mo? | What did you say, "yes"? | Question\r\n'
            'Ako Me Pronoun\r\n'
        )
        args = [*self.args, "--sentences-output", str(self.sentences), "--quiet"]
        self.assertEqual(nmt.main(args), 0)
        expected = [
            list(nmt.FIELDS), ["Ako", "I", "Pronoun"],
            ["Ano an sabi mo?", 'What did you say, "yes"?', "Question"],
            ["Ako", "Me", "Pronoun"],
        ]
        self.assertEqual(self.rows(self.full), expected)
        self.assertEqual(self.rows(self.sentences), [expected[0], expected[2]])
        self.assertEqual(self.rows(self.review), [list(nmt.REVIEW_FIELDS)])
        self.assertIn("Category conflicts: 1", self.report.read_text(encoding="utf-8"))
        paths = (self.full, self.review, self.report, self.sentences, self.source)
        initial = [path.read_bytes() for path in paths]
        self.assertEqual(nmt.main(args), 0)
        self.assertEqual([path.read_bytes() for path in paths], initial)
        for raw in initial[:-1]:
            self.assertNotIn(b"\r", raw)
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(nmt.main([*self.args, "--sentences-only", "--quiet"]), 0)
        self.assertEqual(self.rows(self.full), [expected[0], expected[2]])

    def test_unique_override_strict_dry_run_and_path_guards(self):
        self.write_source("Unresolved text\nUnresolved text\n")
        overrides = self.root / "overrides.csv"
        with overrides.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("match_text", *nmt.FIELDS, "source_line"))
            writer.writerow(("Unresolved text", "Libro", "Book", "Noun", ""))

        args = [*self.args, "--overrides", str(overrides), "--strict"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(nmt.main([*args, "--dry-run"]), 3)
        self.assertIn("found 2 matches", output.getvalue())
        self.assertFalse(any(path.exists() for path in (self.full, self.review, self.report)))
        with overrides.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(("Unresolved text", "Libro", "Book", "Noun", "2"))
        self.assertEqual(nmt.main([*args, "--quiet"]), 3)
        self.assertEqual(self.rows(self.full)[1:], [["Libro", "Book", "Noun"]])
        self.assertEqual(self.rows(self.review)[1][-1], "1")
        self.write_source("Ako I Pronoun\n" * 20 + "Unresolved text\n")
        self.assertEqual(nmt.main([*self.args, "--strict", "--quiet"]), 2)
        raw = self.source.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(nmt.main([*self.args, "--output", str(self.source)]), 1)
        self.assertIn("aliases another file", errors.getvalue())
        self.assertEqual(self.source.read_bytes(), raw)
