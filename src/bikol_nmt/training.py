import csv
from importlib.metadata import version
import json
import math
import platform
import random
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import sacrebleu
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import digest, read, report
from .model import TransformerNMT, threads
from .text import normalize
from .tokenizer import BOS, EOS, PAD, load, token_ids


def decoded(tokenizer, ids):
    tokens = []
    for token in ids:
        if token == EOS:
            break
        if token not in (PAD, BOS):
            tokens.append(token)
    return tokenizer.decode(tokens)


class TranslationDataset:
    def __init__(self, rows, tokenizer, limit):
        self.items = [(token_ids(tokenizer, en), token_ids(tokenizer, bk)) for bk, en, _ in rows]
        if not self.items or any(max(len(src), len(tgt)) > limit for src, tgt in self.items):
            raise ValueError("Dataset is empty or contains an overlong sequence")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        source, target = self.items[index]
        return source, target[:-1], target[1:]


def collate(items):
    return tuple(nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in column], batch_first=True,
                                          padding_value=PAD) for column in zip(*items))


def save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(config, train_path, val_path, tokenizer_path, output, resume=False, dry_run=False):
    settings = config["training"]
    seed = config["seed"]
    threads(settings["threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = load(tokenizer_path)
    train_rows, val_rows = read(train_path), read(val_path)
    limit = config["model"]["max_len"]
    train_data = TranslationDataset(train_rows, tokenizer, limit)
    val_data = TranslationDataset(val_rows, tokenizer, limit)
    rng = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=True,
                              generator=rng, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=settings["batch_size"], collate_fn=collate, num_workers=0)
    model = TransformerNMT(len(tokenizer), **config["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], betas=(.9, .98),
                                  eps=1e-9, weight_decay=.01)
    total_steps = settings["epochs"] * len(train_loader)
    warmup = min(settings["warmup_steps"], max(1, total_steps // 5))
    floor = 1e-5 / settings["learning_rate"]

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1, (step - warmup) / max(total_steps - warmup, 1))
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=settings["label_smoothing"], reduction="sum")
    hashes = {"train": digest(train_path), "val": digest(val_path), "tokenizer": digest(tokenizer_path)}
    if dry_run:
        source, target, _ = next(iter(train_loader))
        print(f"Train {len(train_data)}, validation {len(val_data)}, logits {tuple(model(source, target).shape)}, parameters {sum(p.numel() for p in model.parameters()):,}")
        return
    versions = {name: version(name) for name in ("torch", "numpy", "sentencepiece", "sacrebleu")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False) if shutil.which("git") else None
    metadata = {"config": config, "package_versions": versions, "dataset_hashes": hashes,
                "tokenizer_path": str(tokenizer_path), "vocab_size": len(tokenizer),
                "command": sys.argv, "hardware": platform.platform(), "processor": platform.processor(),
                "git_commit": git.stdout.strip() if git and git.returncode == 0 else None}
    output.mkdir(parents=True, exist_ok=True)
    log = output / "training_log.csv"
    first = 1
    stale = 0
    best = (float("-inf"), float("-inf"))
    if resume:
        state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != config or state["dataset_hashes"] != hashes:
            raise ValueError("Resume requires the original configuration, splits and tokenizer")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        torch.set_rng_state(state["torch_rng"])
        rng.set_state(state["loader_rng"])
        first, stale, best = state["epoch"] + 1, state["stale"], tuple(state["best_metric"])
        with log.open(encoding="utf-8", newline="") as stream:
            history = list(csv.reader(stream))
        with log.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(history[:first])
    report(output / "config.json", [json.dumps(metadata, indent=2)])
    with log.open("a" if resume else "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        if not resume:
            writer.writerow(("epoch", "train_loss", "perplexity", "val_loss", "val_bleu", "val_chrf", "learning_rate", "elapsed_s"))
        for epoch in range(first, settings["epochs"] + 1):
            if stale >= settings["early_stopping_patience"]:
                break
            started = perf_counter()
            model.train()
            total, tokens = 0., 0
            for source, target, labels in train_loader:
                optimizer.zero_grad(set_to_none=True)
                count = labels.ne(PAD).sum().item()
                loss = loss_fn(model(source, target).flatten(0, 1), labels.flatten())
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite")
                (loss / count).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                total += loss.item()
                tokens += count
            model.eval()
            val_loss, val_tokens, hypotheses = 0., 0, []
            with torch.inference_mode():
                for source, target, labels in val_loader:
                    val_loss += loss_fn(model(source, target).flatten(0, 1), labels.flatten()).item()
                    val_tokens += labels.ne(PAD).sum().item()
                    hypotheses.extend(decoded(tokenizer, ids) for ids in model.greedy_decode(source).tolist())
            references = [[normalize(row[0]) for row in val_rows]]
            bleu = sacrebleu.corpus_bleu(hypotheses, references, tokenize="none").score
            chrf = sacrebleu.corpus_chrf(hypotheses, references).score
            validation = val_loss / val_tokens
            ranking = (chrf, -validation)
            improved = ranking > best
            best = ranking if improved else best
            stale = 0 if improved else stale + 1
            state = dict(metadata, model_state_dict=model.state_dict(), epoch=epoch, stale=stale,
                         best_metric=best)
            if improved:
                save(output / "best.pt", state)
            state.update(optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                         torch_rng=torch.get_rng_state(), loader_rng=rng.get_state())
            elapsed = perf_counter() - started
            writer.writerow((epoch, total / tokens, math.exp(total / tokens), validation, bleu, chrf,
                             optimizer.param_groups[0]["lr"], elapsed))
            stream.flush()
            save(output / "last.pt", state)
            report(output / "metrics.json", [json.dumps({"epoch": epoch, "best_val_chrf": best[0],
                                                       "best_val_loss": -best[1]}, indent=2)])
            print(f"Epoch {epoch}: loss {total/tokens:.3f}, val loss {validation:.3f}, chrF {chrf:.2f}, {elapsed:.1f}s",
                  flush=True)
