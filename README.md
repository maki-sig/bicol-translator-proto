# Offline English-to-Bikol-Naga translator

Run from the repository root in PowerShell. With the local artifacts in place:

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python scripts/translate.py --interactive
& $python scripts/translate.py --text 'Good morning.'
& $python scripts/translate.py --method nmt --text 'My friend is studying at home.'
& $python scripts/translate.py --allow-low-confidence --text 'My friend is studying at home.'
& $python scripts/translate.py --input-file inputs.txt --output-file outputs.tsv
```

Use `:help` in interactive mode or `--json` for machine-readable results.

## How translation works

The translator has two ways to answer. It looks for a translation in the
project's dictionary first. If the sentence is unfamiliar, it asks the selected
translation model to write a Bikol version.

```text
English input → Translation controller → Look up the full input in SQLite
                                          ├─ Match found → Return saved translation
                                          └─ No match    → Ask the selected model
                                                              ↓
                            Show the translation, alternatives, method, and time
```

1. **Read the English.** The terminal accepts a single line, repeated inputs
   in an interactive session, or a file with one input per line.
2. **Check the whole input.** For the lookup, the translator ignores
   differences such as capitalization, extra spaces, and a final period.
   For example, “Good morning.” can match the saved entry for “good morning”.
   If the dictionary has several translations, the terminal can show them as
   alternatives.
3. **Generate a translation when needed.** MarianMT is the default model.
   It breaks the English into smaller pieces so it can use the full sentence
   as context while predicting the Bikol text. The custom
   Transformer is also available with `--engine custom` or `:engine custom`.
4. **Show the result.** The terminal reports whether the answer came from the
   dictionary or a model. A dictionary answer uses the wording saved in the
   dataset. A model answer is a prediction and can miss details or change
   the meaning.

## How the project is built

| Part | What it does |
|---|---|
| [Data preparation](scripts/build_nmt_csv.py) | Turns the raw dataset into a clean table of English and Bikol pairs. The full table feeds the dictionary, and the sentence table is split for custom model training and evaluation. |
| [SQLite dictionary](scripts/build_dictionary.py) | Stores known translations in a local database for fast, exact lookup. |
| [Translation controller](src/bikol_nmt/inference.py) | Checks the dictionary and loads the selected model when it is first needed. Both models return results in the same format for the terminal. |
| [Translation models](src/bikol_nmt/marian.py) | MarianMT was trained by Helsinki-NLP and downloaded to this computer. The [custom Transformer](src/bikol_nmt/model.py) was trained on this project's own sentence pairs. Both run locally on the CPU. |
| [Terminal interface](src/bikol_nmt/cli.py) | Handles commands, engine switching, and output for interactive, one-shot, and batch use. |

MarianMT needs an internet connection for initial setup. Once its files and
dependencies are installed, translation runs offline. Dictionary matches can
be returned before either model is loaded.

## MarianMT engine

The project also supports Helsinki-NLP's English-to-Central-Bikol OPUS-MT
checkpoint. Setup needs internet access once. The downloader verifies each
file against the pinned Hugging Face revision and stores the roughly 302 MB
snapshot in `artifacts/marian/opus-mt-en-bcl`. [Upstream model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-bcl)

```powershell
& $python -m pip install -r requirements-marian.txt
& $python scripts/download_marian.py
& $python scripts/translate.py --interactive
& $python scripts/translate.py --text 'My friend is studying at home.'
& $python scripts/translate.py --engine custom --interactive
& $python scripts/benchmark_nmt.py
```

Once installed, MarianMT loads only from the local snapshot and runs in
float32 on CPU. Use `:engine custom` or `:engine marian` to switch during a
session, `:model` to inspect the active engine, and `--marian-dir` to use a
snapshot stored elsewhere. Custom confidence cutoffs and experimental
`--quantize` apply to the custom engine.

The benchmark uses the same held-out CSV and scoring normalization for both
engines. It measures raw NMT predictions with the dictionary bypassed, one
CPU thread, and beam size five by default. Model loading time is recorded
separately from warm median and p95 inference time. Peak memory is measured
in a fresh process for each model. See `artifacts/benchmark/metrics.json`
and `predictions.csv`. The split was held out from custom training. Its
overlap with MarianMT's upstream training corpus is unknown.

### Terminal and scripting options

```text
English    Good morning.
Bikol      Marhay na aga po.
Also       Marhay na aga
Method     dictionary · Greeting
Time       0.5 ms
```

- Use `--quiet` for just the primary translation or `--plain` for ASCII
  `key=value` lines. Plain output escapes non-ASCII input and translation
  characters so their values can be recovered.
- Use `--details` (also `--verbose`) to see available model scores, token counts,
  and the first-load time. NMT badges reflect the calibrated cutoff.
- Use `--no-color`, `--ascii`, or `--width 40` to adjust the display.
  Color defaults to automatic and respects `NO_COLOR`. `--color always`
  requests color in redirected text.
- Interactive commands include `:method dict`, `:beam 5`, `:topk 3`,
  `:timing off`, `:low on`, `:color off`, and `:json on`. Use
  `:config`, `:model`, `:stats`, and `:history` to inspect the session.
  `:help beam` gives command-specific help. History stays in the session.
- One-shot `--json` writes one object. `--json-pretty` indents it. The
  existing keys are preserved, with optional numeric `nll` and `confidence`
  fields on NMT candidates. Interactive JSON writes one object per line to
  stdout and puts shell messages on stderr.

Batch mode defaults to TSV on stdout or in the selected output file. Use
`--format text` for readable blocks or `--format jsonl` for a stream of
JSON objects. The TSV retains its original columns and candidate objects.
Use `--skip-blank` to omit blank lines and `--fail-fast` to stop on the
first runtime error. Progress and the final summary go to stderr. A declined
translation is a successful request with method `no_match`.

```powershell
& $python scripts/translate.py --plain --text 'Good morning.'
& $python scripts/translate.py --json-pretty --text 'Good morning.'
& $python scripts/translate.py --input-file inputs.txt --format text --width 40
& $python scripts/translate.py --input-file inputs.txt --format jsonl --output-file results.jsonl
```

Exit codes are `0` for success, `1` for a runtime error, `2` for an invalid
option, and `130` for an interrupt. Use `--debug` to include tracebacks.

## Rebuild from the raw dataset

The tested Windows environment uses Python 3.15 and the CPU-only PyTorch
distribution in `.venv`. If you need to create that environment, install the
dependencies from a local wheelhouse containing their transitive dependencies.

```powershell
python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install --no-index --find-links .\wheelhouse -r requirements.txt
```

```powershell
& $python scripts/build_nmt_csv.py --overrides data/manual_overrides.csv --sentences-output data/processed/bikol_naga_sentences.csv
& $python scripts/verify_dataset.py
& $python scripts/split_dataset.py
& $python scripts/preprocess.py
& $python scripts/train_tokenizer.py
& $python scripts/build_dictionary.py
& $python scripts/build_dictionary.py --input data/splits/train.csv --output data/dictionary/bikol_naga_train_only.sqlite
& $python scripts/train_nmt.py --config configs/cpu_small.json
& $python scripts/calibrate_nmt.py
& $python scripts/evaluate_nmt.py
& $python -m unittest discover -s tests -v
```

Custom pipeline execution, benchmarking, and translation run locally. Tokenizer training uses
only the training texts. Byte fallback preserves characters outside the learned
BPE vocabulary. The splitter groups normalized English alternatives and
near-duplicate pairs before making the seed-42 assignment. The validation
checkpoint is selected by chrF with validation loss as a tie-breaker. Use
`--dry-run` to check model dimensions or `--resume` to continue the original
training configuration and inputs from `last.pt`.

## Evaluating translations

See `artifacts/eval/quality_audit.txt` for the tokenizer comparison and cutoff
measurements. See `artifacts/eval/metrics.json` for held-out scores and bootstrap intervals,
`test_predictions.csv` and `error_analysis.txt` for examples, and
`latency_report.txt` for the thread and beam benchmarks. Evaluation uses the
dictionary built from the training split. Production dictionary coverage includes
test sentences and is reported separately.

On the i5-1235U, the 111-pair test run reached 35.57 BLEU and 54.11 chrF.
The leakage-controlled test had zero train-dictionary hits, so the fair hybrid
and NMT scores were identical. In the predictions, the model sometimes dropped
time qualifiers or substituted an unrelated location or action from a familiar
sentence pattern. For example, it dropped “this afternoon” from one question
and replaced borrowing a bag with renting a vehicle in another.

The model trained on fewer than a thousand pairs. Unfamiliar English can produce
literal, incomplete, or incorrect Bikol, even with a high beam score. The
calibration command uses validation scores and the English vocabulary from
training to set cutoffs for the current checkpoint. With those cutoffs, the CLI
can decline an unfamiliar sentence. Use `--allow-low-confidence` to inspect the
model's suggestions.
The evaluation report gives full-set scores and the cutoff's coverage and
mean sentence chrF on returned test predictions. Beam scores are
length-adjusted log probabilities. The dictionary keeps the dataset's casing
and variant spellings. The optional `--quantize` and `--threads` switches let you
experiment with CPU inference. The default uses floating-point weights and one
inference thread.
