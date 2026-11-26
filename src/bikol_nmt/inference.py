from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from time import perf_counter

from .dictionary import Dictionary
from .text import display, fold, lookup_key

ENGINES = ("custom", "marian")

def english_words(text):
    return re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*|\d+", fold(text))


@dataclass
class TranslationCandidate:
    text: str
    score: float | None
    source: str
    category: str | None = None
    nll: float | None = None
    confidence: float | None = None
    token_count: int | None = field(default=None, repr=False, compare=False)
    below_cutoff: bool = field(default=False, repr=False, compare=False)
    engine: str | None = field(default=None, compare=False)


@dataclass
class TranslationResult:
    source_text: str
    translations: list[TranslationCandidate]
    method: str
    elapsed_ms: float
    warnings: list[str]


class NeuralTranslator:
    def __init__(self, model_path, tokenizer_path, config):
        import torch
        from .data import digest
        from .model import TransformerNMT, threads
        from .tokenizer import load

        self.torch = torch
        self.tokenizer = load(tokenizer_path)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.checkpoint_epoch = state.get("epoch")
        self.training_hash = state["dataset_hashes"]["train"]
        self.validation_hash = state["dataset_hashes"]["val"]
        if state["dataset_hashes"]["tokenizer"] != digest(tokenizer_path) or state["vocab_size"] != len(self.tokenizer):
            raise ValueError("Checkpoint and tokenizer do not match")
        self.settings = state["config"]["inference"] | config
        self.max_token_nll = None
        self.source_vocabulary = frozenset()
        self.max_unseen_words = None
        policy_path = Path(model_path).with_name("quality_policy.json")
        if policy_path.is_file():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            if policy.get("checkpoint_sha256") == digest(Path(model_path)):
                cutoff = policy.get("max_mean_token_nll")
                if cutoff is not None and (not isinstance(cutoff, (int, float)) or not 0 < cutoff < float("inf")):
                    raise ValueError("Invalid mean token loss cutoff in quality_policy.json")
                self.max_token_nll = cutoff
                self.source_vocabulary = frozenset(policy.get("training_english_vocabulary", ()))
                self.max_unseen_words = policy.get("max_unseen_words")
        threads(self.settings["threads"])
        self.model = TransformerNMT(len(self.tokenizer), **state["config"]["model"])
        self.model.load_state_dict(state["model_state_dict"])
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.model.eval()
        if self.settings.get("quantize"):
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def ranked(self, text, top_k, beam_size):
        from .model import length_normalizer
        from .tokenizer import token_ids
        ids = token_ids(self.tokenizer, text)
        if len(ids) > self.model.max_len:
            raise ValueError("Input exceeds the model's token limit")
        hypotheses = self.model.beam_search(
            self.torch.tensor([ids]), beam_size, max(beam_size, top_k),
            self.settings["max_len"], length_penalty=self.settings["length_penalty"],
        )[0]
        results = {}
        for tokens, score in hypotheses:
            text = display(self.tokenizer.decode(list(tokens)))
            if not lookup_key(text) or len(text) > 500 or "\ufffd" in text or "<unk>" in text:
                continue
            words = re.findall(r"\w+(?:[-']\w+)*", text.lower())
            if not words or any(a == b == c for a, b, c in zip(words, words[1:], words[2:])):
                continue
            length = len(tokens) + 1  # EOS contributes to the beam score.
            nll = -score * length_normalizer(length, self.settings["length_penalty"]) / length
            cutoff = self.max_token_nll
            confidence = None if cutoff is None else max(0., min(1., 1. - nll / cutoff))
            candidate = TranslationCandidate(text, score, "nmt", nll=nll, confidence=confidence,
                                             token_count=len(tokens),
                                             below_cutoff=cutoff is not None and nll > cutoff, engine="custom")
            results.setdefault(lookup_key(text), (candidate, nll))
        return list(results.values())

    def translate(self, text, top_k, beam_size):
        return [candidate for candidate, _ in self.ranked(text, top_k, beam_size)[:top_k]]

    def unknown_words(self, text):
        return list(dict.fromkeys(word for word in english_words(text) if word not in self.source_vocabulary))

    def limited_vocabulary(self, text):
        return self.max_unseen_words is not None and len(self.unknown_words(text)) > self.max_unseen_words

    def reliable(self, ranked, source):
        if self.limited_vocabulary(source):
            return []
        return [candidate for candidate, nll in ranked
                if self.max_token_nll is None or nll <= self.max_token_nll]


class Translator:
    def __init__(self, dictionary_path, model_path, tokenizer_path, config=None, *,
                 engine="custom", marian_dir=None):
        if engine not in ENGINES:
            raise ValueError(f"Unknown NMT engine: {engine}")
        self.dictionary = Dictionary(Path(dictionary_path)) if dictionary_path else None
        self.paths = (model_path, tokenizer_path)
        self.engine = engine
        self.marian_dir = marian_dir
        self.config = config or {}
        self.neural = None

    def close(self):
        if self.dictionary:
            self.dictionary.close()
        self.neural = None

    def select_engine(self, engine):
        if engine not in ENGINES:
            raise ValueError(f"Unknown NMT engine: {engine}")
        if engine != self.engine:
            self.neural = None
            self.engine = engine

    def load_neural(self):
        if self.neural is not None:
            return self.neural
        if self.engine == "marian":
            from .marian import MarianTranslator

            self.neural = MarianTranslator(self.marian_dir, self.config)
        else:
            for label, path in zip(("Model checkpoint", "Tokenizer"), self.paths):
                if not path or not Path(path).is_file():
                    raise FileNotFoundError(f"{label} file is missing: {path}")
            self.neural = NeuralTranslator(*self.paths, self.config)
        return self.neural

    def translate(self, text, method="auto", top_k=3, beam_size=5, *, strict_errors=False):
        if method not in {"auto", "dictionary", "nmt"} or not 1 <= top_k <= 50 or not 1 <= beam_size <= 50:
            raise ValueError("Invalid method, top-k or beam size")
        started = perf_counter()
        warnings = []
        translations = []
        used = "no_match"
        if not lookup_key(text):
            warnings.append("Enter English text to translate.")
        else:
            if method != "nmt" and self.dictionary:
                translations = [TranslationCandidate(bikol, None, "dictionary", category)
                                for bikol, category in self.dictionary.lookup(text)[:top_k]]
                if translations:
                    used = "dictionary"
            if not translations and method != "dictionary":
                try:
                    neural = self.load_neural()
                    allow_low = neural.settings.get("allow_low_confidence")
                    unfamiliar = neural.limited_vocabulary(text)
                    if unfamiliar:
                        words = ", ".join(neural.unknown_words(text)[:6])
                        warnings.append(f"Several English words are outside the model's training vocabulary: {words}.")
                    ranked = neural.ranked(text, top_k, beam_size) if allow_low or not unfamiliar else []
                    reliable = neural.reliable(ranked, text)
                    if allow_low:
                        translations = [candidate for candidate, _ in ranked[:top_k]]
                        if unfamiliar and translations:
                            warnings.append("Check the meaning of these model guesses before use.")
                        elif any(candidate not in reliable for candidate in translations):
                            warnings.append("The model scored one or more translations below the cutoff. Check their meaning before use.")
                    else:
                        translations = reliable[:top_k]
                        if unfamiliar:
                            warnings.append("Use --allow-low-confidence to inspect a model guess.")
                        elif ranked and not translations:
                            warnings.append("The model scored all translations below the cutoff. Use --allow-low-confidence to inspect them.")
                    if translations:
                        used = "nmt"
                except (OSError, ValueError, RuntimeError, ImportError) as error:
                    if strict_errors:
                        raise
                    warnings.append(str(error))
            if not translations and not warnings:
                warnings.append("No translation found.")
        return TranslationResult(text, translations, used, (perf_counter() - started) * 1000, warnings)
