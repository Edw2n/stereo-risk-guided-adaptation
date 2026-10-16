# Additional paper analyses

## Recalculate the completed comparisons

```bash
python reproduce.py paper report --run-id paper
```

Outputs include `native_metrics.csv`, `adaptation.tex`,
`signal_vs_baselines.csv`, `clean_preservation.csv` and
`dev_candidate_ranking.csv`. The command uses the same 328 development and
172 held-out pairs as the paper. Fixed-best is one candidate shared by both
models and all conditions: minimize the mean of model-wise pooled development
D1, then Abs, then 1px. Evaluation data never enter this choice.

Sequence-cluster intervals resample entire sequences with common weights across
methods and assignments. Assignment standard deviation is reported separately.
The point estimate for recovery is the mean of per-assignment ratios.

## Additional inference

Only run these commands when new predictions are needed. They use the existing
frozen models and bank and write a new output directory.

```bash
python reproduce.py paper revision prepare --run-id additional   --dev-assignment configs/paper/assignments/dev328.json
python reproduce.py paper revision infer --run-id additional --stage fixed-dev --model raft --batch 2
python reproduce.py paper revision infer --run-id additional --stage fixed-dev --model acv --batch 4
python reproduce.py paper revision shared-fixed-best --run-id additional --output-tag shared-fixed
```

Repeat the inference command with `--stage clean` or `--stage denoise` for each
model. The clean stage evaluates the fixed bank on clean inputs; denoising uses
the predeclared Gaussian filter. Explicit batch choices are saved in the run
record. `--limit 1` is a separate smoke run, not a full-result estimate.

## Candidate budget and consensus

```bash
python reproduce.py paper revision offline --run-id additional --stage budget
python reproduce.py paper revision offline --run-id additional --stage conditions
```

These commands use included per-candidate native statistics and exact pairwise
rank counts. They re-rank each subset instead of truncating scores from the
full bank. Candidate budgets include Identity and use the predeclared proposal
seeds 9137, 9138 and 9139.

For consensus score extraction, pass `--stage consensus --evidence DIRECTORY`,
where `DIRECTORY/adaptation/seedSEED/MODEL.jsonl` contains the saved
`pool.path` and `pool.group` fields. A missing pool is an error; no inference
is started to replace it. The supplied compact journals retain the consensus
scores separately but do not include dense predictions or disparity pools.

To report new additional runs, use:

```bash
python reproduce.py paper revision report --run-id additional --output-tag comparisons   --evidence DIRECTORY   --shared-fixed-best "$ROCO_ROOT/outputs/additional/shared-fixed/SHARED_FIXED_BEST.json"
```

This requires the completed stage journals and their clean disparity pools.
For the distributed evidence, `paper report` is the CPU-only entry point.
