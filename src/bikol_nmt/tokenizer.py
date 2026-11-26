import io

import sentencepiece as spm

from .data import digest, report
from .text import normalize

PAD, UNK, BOS, EOS = range(4)


def token_ids(tokenizer, text):
    return [BOS, *tokenizer.encode(normalize(text)), EOS]


def load(path):
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    if (tokenizer.pad_id(), tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()) != (PAD, UNK, BOS, EOS):
        raise ValueError("Unexpected tokenizer special tokens")
    return tokenizer


def train(en, bk, output, vocab_size=4000, coverage=.9995):
    parameters = dict(model_type="bpe", vocab_size=vocab_size, character_coverage=coverage,
                      pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, input_sentence_size=100000,
                      shuffle_input_sentence=False, num_threads=8, hard_vocab_limit=False,
                      normalization_rule_name="identity", byte_fallback=True)
    corpora = [path.read_text(encoding="utf-8").splitlines() for path in (en, bk)]
    if len(corpora[0]) != len(corpora[1]) or not corpora[0] or not all(map(all, corpora)):
        raise ValueError("Tokenizer needs aligned nonempty training text files")
    binary = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=(text for corpus in corpora for text in corpus),
                                  model_writer=binary, minloglevel=2, **parameters)
    output.mkdir(parents=True, exist_ok=True)
    model = output / "bikol_naga_bpe.model"
    model.write_bytes(binary.getvalue())
    tokenizer = load(model)
    report(output / "bikol_naga_bpe.vocab",
           [f"{tokenizer.id_to_piece(i)}\t{tokenizer.get_score(i)}" for i in range(len(tokenizer))])
    lines = [str(parameters), f"Vocabulary: {len(tokenizer)}", f"SentencePiece: {spm.__version__}",
             f"Model SHA-256: {digest(model)}", *(f"{p} SHA-256: {digest(p)}" for p in (en, bk))]
    for corpus in corpora:
        for text in corpus[:5]:
            if tokenizer.decode(tokenizer.encode(text)) != text:
                raise ValueError(f"Tokenizer round trip failed for {text!r}")
    for split in ("train", "val", "test"):
        for lang, source in (("en", en), ("bk", bk)):
            path = source if split == "train" else source.with_name(f"{split}.{lang}.txt")
            if path.exists():
                encoded = tokenizer.encode(path.read_text(encoding="utf-8").splitlines())
                total = sum(map(len, encoded))
                unknown = sum(tokens.count(UNK) for tokens in encoded)
                lines.append(f"{split}.{lang}: OOV {unknown/max(total, 1):.2%}, mean subwords {total/max(len(encoded), 1):.2f}")
    report(output / "tokenizer_report.txt", lines)
