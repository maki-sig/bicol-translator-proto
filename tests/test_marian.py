import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bikol_nmt import data, dictionary
from bikol_nmt.inference import Translator
from bikol_nmt.marian import LOCAL_FILES, MarianTranslator


class MarianTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_switching_preserves_dictionary_first_with_unavailable_models(self):
        csv = self.root / "entries.csv"
        db = self.root / "entries.sqlite"
        data.write(csv, data.FIELDS, [("Marhay na aga.", "Good morning.", "Greeting")])
        dictionary.build(csv, db)
        translator = Translator(db, None, None, engine="marian", marian_dir=self.root / "missing")
        self.addCleanup(translator.close)
        self.assertEqual(translator.translate("Good morning!").method, "dictionary")
        self.assertIsNone(translator.neural)
        translator.neural = object()
        translator.select_engine("custom")
        self.assertIsNone(translator.neural)
        self.assertEqual(translator.translate("Good morning!").method, "dictionary")
        translator.select_engine("marian")
        with self.assertRaisesRegex(FileNotFoundError, "MarianMT files missing"):
            translator.translate("A new sentence.", strict_errors=True)
        run = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/translate.py"), "--interactive",
             "--dictionary", str(db), "--marian-dir", str(self.root / "missing"), "--json"],
            input=":config\nGood morning!\n:engine custom\nGood morning.\n:q\n",
            capture_output=True, encoding="utf-8", check=True,
        )
        self.assertEqual([json.loads(line)["method"] for line in run.stdout.splitlines()],
                         ["dictionary", "dictionary"])
        self.assertRegex(run.stderr, r"Engine\s+marian")
        self.assertIn("NMT engine set to custom.", run.stderr)

    def test_marian_loads_from_local_files_with_cpu_decode(self):
        for name in LOCAL_FILES:
            (self.root / name).touch()
        tokenizer = Mock(eos_token_id=0, unk_token_id=1, all_special_ids=[0, 1, 9])
        tokenizer.return_value = {"input_ids": torch.tensor([[8, 0]])}
        tokenizer.decode.return_value = "nag - aadal ako."
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.parameters.return_value = [torch.empty(2)]
        model.config.max_position_embeddings = 32
        model.generate.return_value = SimpleNamespace(sequences=torch.tensor([[9, 8, 0]]))
        hf = SimpleNamespace(MarianTokenizer=Mock(), MarianMTModel=Mock())
        hf.MarianTokenizer.from_pretrained.return_value = tokenizer
        hf.MarianMTModel.from_pretrained.return_value = model
        with patch.dict(sys.modules, {"transformers": hf}), patch.dict("os.environ", clear=False):
            engine = MarianTranslator(self.root, {})
            candidates = engine.ranked("  It’s  my friend. ", 1, 1)
        hf.MarianTokenizer.from_pretrained.assert_called_once_with(self.root.resolve(), local_files_only=True)
        hf.MarianMTModel.from_pretrained.assert_called_once_with(
            self.root.resolve(), local_files_only=True, use_safetensors=False, dtype=torch.float32,
        )
        model.to.assert_called_once_with("cpu")
        tokenizer.assert_called_once_with("It's my friend.", return_tensors="pt", verbose=False)
        self.assertEqual(model.generate.call_args.kwargs["num_beams"], 1)
        self.assertNotIn("length_penalty", model.generate.call_args.kwargs)
        self.assertEqual(candidates[0][0].text, "Nag-aadal ako.")
        self.assertEqual(candidates[0][0].engine, "marian")
        self.assertIsNone(candidates[0][0].confidence)
