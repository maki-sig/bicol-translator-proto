"""Local CPU inference for the English-to-Central Bikol OPUS-MT snapshot."""

import os
from pathlib import Path
import re

from .inference import TranslationCandidate
from .text import clean_text, display, lookup_key


MODEL_ID = "Helsinki-NLP/opus-mt-en-bcl"
LOCAL_FILES = ("config.json", "generation_config.json", "pytorch_model.bin",
               "source.spm", "target.spm", "tokenizer_config.json", "vocab.json")


class MarianTranslator:
    def __init__(self, directory, config):
        if directory is None:
            raise FileNotFoundError("MarianMT model directory was not provided")
        path = Path(directory).resolve()
        missing = [name for name in LOCAL_FILES if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"MarianMT files missing in {path}: {', '.join(missing)}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        from .model import threads
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as error:
            raise ImportError("MarianMT requires the packages in requirements-marian.txt") from error

        self.settings = {"threads": 1, "max_len": 48, "length_penalty": .6} | config
        self.max_token_nll = None
        self.max_unseen_words = None
        threads(self.settings["threads"])
        self.torch = torch
        self.tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
        self.model = MarianMTModel.from_pretrained(
            path, local_files_only=True, use_safetensors=False, dtype=torch.float32,
        ).to("cpu").eval()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())

    def ranked(self, text, top_k, beam_size):
        # Keep the upstream casing and punctuation before Marian's SentencePiece
        # tokenizer. The shared dictionary lookup still uses its normalized key.
        inputs = self.tokenizer(clean_text(text), return_tensors="pt", verbose=False)
        if inputs["input_ids"].size(1) > self.model.config.max_position_embeddings:
            raise ValueError("Input exceeds the MarianMT token limit")
        beam_options = {"length_penalty": self.settings["length_penalty"]} if beam_size > 1 else {}
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, num_beams=beam_size,
                num_return_sequences=min(top_k, beam_size),
                max_new_tokens=self.settings["max_len"] - 1,
                forced_eos_token_id=None,
                return_dict_in_generate=True, output_scores=beam_size > 1,
                **beam_options,
            )
        translations = {}
        special = set(self.tokenizer.all_special_ids)
        scores = getattr(output, "sequences_scores", None)
        for i, tokens in enumerate(output.sequences):
            tokens = tokens.tolist()
            if self.tokenizer.eos_token_id not in tokens or self.tokenizer.unk_token_id in tokens:
                continue
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            # OPUS can emit spaces around the hyphen in Bikol forms like nag-aadal.
            translation = display(re.sub(r"(?<=\w) +- +(?=\w)", "-", decoded))
            key = lookup_key(translation)
            if not key:
                continue
            candidate = TranslationCandidate(
                translation, float(scores[i]) if scores is not None else None, "nmt",
                token_count=sum(token not in special for token in tokens), engine="marian",
            )
            translations.setdefault(key, (candidate, None))
        return list(translations.values())

    def limited_vocabulary(self, text):
        # The custom model's calibration was derived from a different corpus.
        return False

    def reliable(self, ranked, source):
        return [candidate for candidate, _ in ranked]
