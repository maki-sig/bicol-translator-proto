from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bikol_nmt import data, dictionary, text, tokenizer, training
from bikol_nmt.inference import NeuralTranslator, TranslationCandidate, Translator
from bikol_nmt.model import BOS, EOS, UNK, TransformerNMT


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_normalization_and_display(self):
        self.assertEqual(text.normalize("  Ta’no, mag-adal?  "), "ta'no , mag-adal ?")
        self.assertEqual(text.lookup_key("  “Good   morning!” "), "good morning")
        self.assertEqual(text.display("marhay , tabi !"), "Marhay, tabi!")

    def test_split_keeps_near_duplicates_and_alternatives_together(self):
        rows = [
            ("Marhay na aga sa gabos.", "Good morning to everyone.", "Greeting"),
            ("Marhay na aga sa gabos!", "Good morning to everyone!", "Greeting"),
            ("Dios marhay na aga.", "Good morning to everyone.", "Greeting"),
            ("Mabalos.", "Thank you.", "Politeness"),
        ]
        first, groups = data.split(rows, 42, (.8, .1, .1), .9)
        self.assertEqual(first, data.split(rows, 42, (.8, .1, .1), .9)[0])
        self.assertIn([0, 1, 2], groups)
        self.assertTrue(any(set(indices) >= {0, 1, 2} for indices in first.values()))

    def test_dictionary_inference_and_cli(self):
        source = self.root / "source.csv"
        db = self.root / "dictionary.sqlite"
        data.write(source, data.FIELDS, [
            ("Marhay na aga", "Good morning", "Phrase"),
            ("Dios marhay na aga.", "Good morning.", "Greeting"),
        ])
        dictionary.build(source, db)
        translator = Translator(db, None, None)
        self.addCleanup(translator.close)
        result = translator.translate("GOOD MORNING!")
        self.assertEqual([x.text for x in result.translations], ["Dios marhay na aga.", "Marhay na aga"])
        self.assertEqual(result.method, "dictionary")
        self.assertGreaterEqual(result.elapsed_ms, 0)
        self.assertIsNone(translator.neural)
        self.assertEqual(translator.translate(" ").method, "no_match")
        self.assertTrue(translator.translate("New input").warnings)
        command = [sys.executable, "-B", "scripts/translate.py", "--dictionary", str(db), "--json"]
        test = subprocess.run([*command, "--text", "GOOD MORNING!"], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(test.stdout)["method"], asdict(result)["method"])
        test = subprocess.run([*command, "--text", ""], text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(test.stdout)["method"], "no_match")

    def test_model_masks_and_beam_search(self):
        training.threads(1)
        model = TransformerNMT(12, 16, 4, 1, 1, 32, 0., 16).eval()
        with torch.inference_mode():
            source = torch.tensor([[BOS, 5, EOS]])
            padded = torch.tensor([[BOS, 5, EOS, 0]])
            target = torch.tensor([[BOS, 6, 7]])
            changed = torch.tensor([[BOS, 6, 8]])
            logits = model(source, target)
            self.assertEqual(tuple(logits.shape), (1, 3, 12))
            torch.testing.assert_close(logits, model(padded, target))
            torch.testing.assert_close(logits[:, :2], model(source, changed)[:, :2])

        class ScriptedModel(TransformerNMT):
            def decode(self, trg, memory, source_mask, last_only=False):
                logits = torch.full((len(trg), 12), -10.)
                if trg.size(1) == 1:
                    logits[:, 5], logits[:, 6] = 4., 3.
                else:
                    logits[:, EOS] = 5.
                return logits

        model = ScriptedModel(12, 16, 4, 1, 1, 32, 0., 16).eval()
        result = model.beam_search(torch.tensor([[BOS, 5, EOS], [BOS, 6, EOS]]),
                                   beam_size=2, top_k=2, max_len=8)
        self.assertEqual([[tokens for tokens, _ in row] for row in result], [[(5,), (6,)], [(5,), (6,)]])
        self.assertGreater(result[0][0][1], result[0][1][1])

        class UnfinishedModel(ScriptedModel):
            def decode(self, trg, memory, source_mask, last_only=False):
                logits = super().decode(trg, memory, source_mask, last_only)
                logits[:, UNK] = 10.
                logits[:, EOS] = -torch.inf
                return logits

        unfinished = UnfinishedModel(12, 16, 4, 1, 1, 32, 0., 16).eval()
        source = torch.tensor([[BOS, 5, EOS]])
        self.assertEqual(unfinished.beam_search(source, beam_size=2, max_len=4), [[]])
        self.assertNotIn(UNK, unfinished.greedy_decode(source, max_len=4).tolist()[0])

    def test_low_confidence_predictions_require_opt_in(self):
        translator = Translator(None, None, None)
        neural = NeuralTranslator.__new__(NeuralTranslator)
        neural.settings = {}
        neural.max_token_nll = .8
        neural.max_unseen_words = None
        candidate = TranslationCandidate("Magtabang kita.", -3.0, "nmt")
        neural.ranked = lambda *_: [(candidate, 1.4)]
        translator.neural = neural
        result = translator.translate("A new sentence.")
        self.assertEqual(result.method, "no_match")
        self.assertIn("below the cutoff", result.warnings[0])
        neural.settings["allow_low_confidence"] = True
        result = translator.translate("A new sentence.")
        self.assertEqual(result.translations, [candidate])
        self.assertIn("below the cutoff", result.warnings[0])
        neural.settings.clear()
        neural.max_unseen_words = 1
        neural.source_vocabulary = {"a"}
        neural.ranked = lambda *_: self.fail("Unsupported inputs should skip beam search")
        result = translator.translate("A new sentence.")
        self.assertEqual(result.method, "no_match")
        self.assertIn("new, sentence", result.warnings[0])

    def test_tokenizer_and_checkpoint_smoke(self):
        en, bk = self.root / "train.en.txt", self.root / "train.bk.txt"
        rows = [("Marhay na aga.", "Good morning.", "Greeting"),
                ("Dios mabalos.", "Thank you.", "Politeness")] * 10
        data.report(en, [text.normalize(row[1]) for row in rows])
        data.report(bk, [text.normalize(row[0]) for row in rows])
        tokenizer.train(en, bk, self.root, 320)
        model_file = self.root / "bikol_naga_bpe.model"
        tokens = tokenizer.load(model_file)
        sample = "ta'no ñ"
        self.assertEqual(tokens.decode(tokens.encode(sample)), sample)
        source, val = self.root / "train.csv", self.root / "val.csv"
        data.write(source, data.FIELDS, rows)
        data.write(val, data.FIELDS, [("Marhay na hapon.", "Good afternoon.", "Greeting")])
        config = {"seed": 42, "model": {"d_model": 16, "nhead": 4, "num_encoder_layers": 1,
                                        "num_decoder_layers": 1, "dim_feedforward": 32, "max_len": 32},
                  "training": {"threads": 1, "batch_size": 4, "epochs": 1, "learning_rate": .001,
                               "warmup_steps": 2, "label_smoothing": .1, "early_stopping_patience": 2},
                  "inference": {"threads": 1, "max_len": 12, "length_penalty": .6}}
        training.train(config, source, val, model_file, self.root)
        training.train(config, source, val, model_file, self.root, resume=True)
        checkpoint = self.root / "best.pt"
        translator = Translator(None, checkpoint, model_file)
        self.addCleanup(translator.close)
        self.assertEqual(translator.load_neural().training_hash, data.digest(source))
        self.assertTrue((self.root / "last.pt").is_file())
        result = translator.translate("Good afternoon", "nmt")
        self.assertEqual(result.source_text, "Good afternoon")
        self.assertGreaterEqual(result.elapsed_ms, 0)
