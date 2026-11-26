from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikol_nmt import cli, data, dictionary, ui
from bikol_nmt.inference import TranslationCandidate, TranslationResult


def result(*candidates, source="Good morning.", method="dictionary", warnings=()):
    return TranslationResult(source, list(candidates), method, 1.2, list(warnings))


class Terminal(io.StringIO):
    def isatty(self):
        return True


class RendererTests(unittest.TestCase):
    def render(self, record, width=100, color=False, unicode=True, **options):
        stream = io.StringIO()
        ui.render(record, ui.Theme(color, unicode, width, stream), **options)
        return stream.getvalue()

    def test_headline_and_unique_alternatives(self):
        first = TranslationCandidate("Marhay na aga.", None, "dictionary", "Greeting")
        second = TranslationCandidate("Marhay na aga po.", None, "dictionary", "Greeting")
        third = TranslationCandidate("Dios marhay na aga.", None, "dictionary", "Greeting")
        self.assertNotIn("Also", self.render(result(first)))
        self.assertIn("Also       Marhay na aga po.", self.render(result(first, second)))
        multiple = self.render(result(first, first, second, third))
        self.assertEqual(multiple.count("Marhay na aga.\n"), 1)
        self.assertIn("1. Marhay na aga po.", multiple)
        self.assertIn("2. Dios marhay na aga.", multiple)
        self.assertIn("dictionary · Greeting", multiple)

    def test_confidence_badges_and_details(self):
        for confidence, below, label in ((.8, False, "high confidence"), (.5, False, "medium confidence"),
                                         (.2, False, "low confidence"), (0., True, "below cutoff")):
            with self.subTest(label=label):
                candidate = TranslationCandidate("Magtabang kita.", -1.23, "nmt", nll=.7,
                                                 confidence=confidence, token_count=4, below_cutoff=below)
                record = result(candidate, method="nmt", warnings=["Check the meaning before use."])
                rendered = self.render(record, details=True, load_ms=420.)
                self.assertIn(label, rendered)
                self.assertIn("score=-1.230 nll=0.700", rendered)
                self.assertIn("tokens=4", rendered)
                self.assertIn("Model load 420.0 ms", rendered)
                self.assertIn("Note       Check the meaning before use.", rendered)

    def test_narrow_wrapping_and_decline_hint(self):
        source = "Words beyond the vocabulary made this sentence difficult."
        warning = "Several English words are outside the model's training vocabulary: quantum, flux."
        record = result(source=source, method="no_match",
                        warnings=[warning, "Use --allow-low-confidence to inspect a model guess."])
        rendered = self.render(record, 40, unicode=False)
        self.assertTrue(rendered.startswith("English\n  "))
        self.assertIn("Bikol\n  --", rendered)
        self.assertIn("Reason\n  ", rendered)
        self.assertIn("--allow-low-confidence", rendered)
        self.assertTrue(all(len(line) <= 40 for line in rendered.splitlines()))
        long = TranslationCandidate("Marhay na pag-umaw, sa gabos na mag-adal nin matanos.", None, "dictionary")
        self.assertIn("mag-adal", self.render(result(long), 40))

    def test_plain_and_quiet(self):
        record = result(TranslationCandidate("Piña", None, "dictionary", "Noun"), source="Pineapple")
        rendered = self.render(record, color=True, output="plain")
        self.assertEqual(rendered, "source=Pineapple\ntranslation=Pi\\xf1a\nmethod=dictionary\n"
                                  "category=Noun\nelapsed_ms=1.2\nwarning=\n")
        self.assertEqual(self.render(record, color=True, output="quiet"), "Piña\n")

    def test_color_and_encoding_capabilities(self):
        sink = io.StringIO()
        self.assertFalse(ui.theme(sink).color)
        self.assertTrue(ui.theme(sink, "always").color)
        self.assertFalse(ui.theme(sink, "always", file_output=True).color)
        with patch.dict("os.environ", {"NO_COLOR": "1"}), patch.object(ui, "supports_ansi", return_value=True):
            self.assertFalse(ui.theme(Terminal()).color)
            self.assertTrue(ui.theme(Terminal(), "always").color)
            self.assertFalse(ui.theme(Terminal(), "never").color)

        class ASCIIStream(io.StringIO):
            encoding = "ascii"

        self.assertFalse(ui.theme(ASCIIStream()).unicode)
        self.assertFalse(ui.theme(sink, ascii_only=True).unicode)
        item = result(TranslationCandidate("Marhay", None, "dictionary"))
        self.assertIn("\033[1;36mMarhay\033[0m", self.render(item, color=True))
        self.assertNotIn("\033", self.render(item))

    def test_non_vt_windows_console_falls_back(self):
        class LegacyTerminal(Terminal):
            def fileno(self):
                return 2

        stream = LegacyTerminal()
        with patch.object(ui.os, "name", "nt"), patch.object(ui, "enable_windows_vt", return_value=False):
            view = ui.theme(stream, color="always")
            self.assertFalse(view.color)
            cli.clear_terminal(view)
        self.assertTrue(stream.getvalue())
        self.assertNotIn("\033", stream.getvalue())


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.db = self.root / "dictionary.sqlite"
        source = self.root / "dictionary.csv"
        data.write(source, data.FIELDS, [
            ("Marhay na aga.", "Good morning.", "Greeting"),
            ("Marhay na aga po.", "Good morning.", "Greeting"),
        ])
        dictionary.build(source, self.db)
        self.inputs = self.root / "inputs.txt"
        self.inputs.write_bytes("Good morning.\r\n\r\nUnlisted sentence.\r\n".encode("utf-8-sig"))

    def run_cli(self, *options, stdin=None):
        script = Path(__file__).resolve().parents[1] / "scripts/translate.py"
        return subprocess.run([sys.executable, "-B", str(script), "--dictionary", str(self.db), *map(str, options)],
                              cwd=self.root, capture_output=True, input=stdin, encoding="utf-8")

    def test_json_contract_and_empty_input(self):
        run = self.run_cli("--json", "--text", "Good morning.")
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads(run.stdout)
        self.assertEqual(set(record), {"source_text", "translations", "method", "elapsed_ms", "warnings"})
        self.assertEqual([x["text"] for x in record["translations"]], ["Marhay na aga.", "Marhay na aga po."])
        self.assertEqual(len(run.stdout.splitlines()), 1)
        empty = self.run_cli("--json-pretty", "--text", "")
        self.assertEqual(json.loads(empty.stdout)["method"], "no_match")
        self.assertGreater(len(empty.stdout.splitlines()), 1)
        nmt = result(TranslationCandidate("Test.", -1.2, "nmt", nll=.7, confidence=.1), method="nmt")
        item = json.loads(ui.result_json(nmt))["translations"][0]
        self.assertEqual((item["nll"], item["confidence"]), (.7, .1))

    def test_interactive_commands_and_json(self):
        feed = "\n:hep\n:HeLp=beam\n:MeThOd = d\n:beam 99\n:beam=4\n:topk 2\n:low on\n:color off\n"
        feed += ":timing off\n:config\n:clear\n:json on\nGood morning.\n:stats\n:history\n:exit\n"
        run = self.run_cli("--interactive", "--ascii", stdin=feed)
        self.assertEqual(run.returncode, 0, run.stderr)
        record = json.loads(run.stdout)
        self.assertEqual(record["method"], "dictionary")
        self.assertEqual(len(record["translations"]), 2)
        for message in ("Did you mean :help?", ":beam 1-50", "1 to 50", "Beam set to 4",
                        "Method set to dictionary", "Timing     off", "Low conf.  on", "Dictionary 1",
                        "Good morning.", "Type an English word"):
            self.assertIn(message, run.stderr)
        self.assertNotIn("\033", run.stderr)
        self.assertEqual(self.run_cli("--interactive", stdin="").returncode, 0)

    def test_live_prompt_and_interrupt(self):
        stdout, stderr = io.StringIO(), Terminal()
        args = ["translate.py", "--interactive", "--dictionary", str(self.db), "--ascii", "--no-color"]
        with patch.object(sys, "argv", args), patch.object(sys, "stdin", Terminal()), redirect_stdout(stdout), (
            redirect_stderr(stderr)
        ), patch("builtins.input", side_effect=[":method dict", KeyboardInterrupt()]):
            self.assertEqual(cli.translate_main(), 130)
        self.assertIn("Bikol-Naga Translator - offline", stderr.getvalue())
        self.assertIn("auto > ", stderr.getvalue())
        self.assertIn("dict > ", stderr.getvalue())

    def test_batch_formats_and_redirected_status(self):
        output = self.root / "output.tsv"
        run = self.run_cli("--method", "dictionary", "--input-file", self.inputs, "--output-file", output)
        self.assertEqual(run.returncode, 0, run.stderr)
        with output.open(encoding="utf-8", newline="") as stream:
            header, *rows = list(csv.reader(stream, delimiter="\t"))
        self.assertEqual(tuple(header), cli.TSV_FIELDS)
        self.assertEqual([row[3] for row in rows], ["dictionary", "no_match", "no_match"])
        candidate = json.loads(rows[0][2])[0]
        self.assertEqual(set(candidate), {"text", "score", "source", "category"})
        self.assertIn("Done", run.stderr)
        self.assertIn("dictionary 1", run.stderr)
        self.assertIn("no match 2", run.stderr)
        self.assertNotIn("Translating", run.stderr)
        implicit = self.run_cli("--method", "dictionary", "--input-file", self.inputs)
        self.assertEqual(tuple(next(csv.reader(io.StringIO(implicit.stdout), delimiter="\t"))), cli.TSV_FIELDS)
        jsonl = self.run_cli("--method", "dictionary", "--input-file", self.inputs, "--format", "jsonl", "--skip-blank")
        self.assertEqual([json.loads(line)["method"] for line in jsonl.stdout.splitlines()], ["dictionary", "no_match"])
        text_file = self.root / "output.txt"
        text = self.run_cli("--method", "dictionary", "--format", "text", "--input-file", self.inputs,
                            "--skip-blank", "--output-file", text_file, "--color", "always")
        self.assertEqual(text.returncode, 0, text.stderr)
        content = text_file.read_text(encoding="utf-8")
        self.assertIn("[1/2]", content)
        self.assertIn("Good morning.", content)
        self.assertIn("Method", content)
        self.assertNotIn("\033", content)

    def test_tty_progress_and_batch_error_policy(self):
        args = cli.TranslateParser()
        cli.translate_options(args)
        options = args.parse_args(["--input-file", str(self.inputs), "--no-color"])

        class FakeTranslator:
            load_ms = None

            def request(self, source):
                if source == "bad":
                    raise ValueError("Source is too long")
                return result(source=source, method="no_match")

        for fail_fast, expected in ((False, ["bad", "after"]), (True, ["bad"])):
            status = Terminal()
            output = io.StringIO()
            options.fail_fast = fail_fast
            with redirect_stderr(status):
                code = cli.batch_rows(FakeTranslator(), ["bad", "after"], options, output, "jsonl")
            self.assertEqual(code, 1)
            self.assertEqual([json.loads(line)["source_text"] for line in output.getvalue().splitlines()], expected)
            self.assertIn("error: Source is too long", status.getvalue())
            if not fail_fast:
                self.assertIn("\rTranslating", status.getvalue())
                self.assertTrue(re.search(r"\r +\rDone", status.getvalue()))
            else:
                self.assertIn("Stopped", status.getvalue())

    def test_error_hints_and_exit_codes(self):
        invalid = self.run_cli("--width", 0, "--text", "Hello")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("error:", invalid.stderr)
        self.assertIn("hint:", invalid.stderr)
        for option, expected in (("--dictionary", "build_dictionary.py"), ("--model", "train_nmt.py"),
                                 ("--tokenizer", "train_tokenizer.py")):
            extra = [] if option == "--dictionary" else ["--method", "nmt", "--engine", "custom"]
            if option == "--tokenizer":
                checkpoint = self.root / "placeholder.pt"
                checkpoint.touch()
                extra += ["--model", checkpoint]
            run = self.run_cli(*extra, option, self.root / "missing", "--text", "Unlisted")
            self.assertEqual(run.returncode, 1)
            self.assertIn(expected, run.stderr)
            self.assertNotIn("Traceback", run.stderr)
        debug = self.run_cli("--dictionary", self.root / "missing", "--debug", "--text", "Unlisted")
        self.assertIn("Traceback", debug.stderr)
