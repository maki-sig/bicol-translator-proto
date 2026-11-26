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

The CLI normalizes the full input for an exact SQLite lookup. On a miss it
uses a Transformer trained from scratch on this project's data. It displays
alternatives, method, and elapsed time. Use `:help` in interactive mode or
`--json` for machine-readable one-shot results. Dictionary-only queries can
run without importing the model dependencies.

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

Pipeline execution is local and makes no network calls. Tokenizer training uses
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
