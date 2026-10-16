# Inference from scratch

Use this workflow to generate new predictions with the frozen paper or
challenge configuration. To reproduce tables from the included results, use
[the CPU reporting commands](../README.md#reproduce-reported-results).

## Prepare the environment and assets

Complete [setup](SETUP.md) first. It configures output paths and installs the
paper and CroCo runtimes. The examples below use `ROCO_ROOT` from that guide;
if using an existing configuration, use its `storage_root` for output paths.
[Data and checkpoints](ASSETS.md) lists download sources, pinned versions,
licenses, and the expected layout.

For the paper:

```bash
for model in raft acv; do
  python reproduce.py assets clone-model --model "$model"
  python reproduce.py assets download-model --model "$model"
done
python reproduce.py assets download-paper
```

For the challenge:

```bash
python reproduce.py assets clone-model --model croco
python reproduce.py assets download-model --model croco
python reproduce.py assets download-official --condition clean
python reproduce.py assets download-official --condition gaussian_noise
```

Download the other official corruption conditions as needed. The paper uses
public Spring train RGB/GT with the fixed 328-pair development and 172-pair
evaluation split. The challenge uses official test inputs; test GT is not
available locally. No model training is required.

## Paper: RAFT-Stereo and ACVNet

### Generate task and adaptation predictions

```bash
python reproduce.py paper phase-a --model raft --split dev --run-id phase-a
python reproduce.py paper phase-a --model acv --split dev --run-id phase-a

for seed in 101 202 303; do
  python reproduce.py paper adaptation --model raft --seed $seed --run-id adaptation-$seed
  python reproduce.py paper adaptation --model acv --seed $seed --run-id adaptation-$seed
done
```

A smoke run uses a separate ID, e.g. `paper phase-a --model raft --split dev --conditions clean --limit 1 --batch 1 --run-id smoke-raft`. Batch overrides are recorded. Production defaults are RAFT B2 / ACV B4, FP32, native 1920×1080. ACV keeps maxdisp=192 and evaluates the full GT domain.

Phase A stores a shared deterministic 50k pixel pool and analyzes its first 20k. Development scalar journals are `<run>/MODEL/samples.jsonl`; held-out journals remain separate under `<run>/artifacts/intermediates/`. Inference checks the required assets before loading a model. Phase A rejects incomplete output directories; preserve them and use a new run ID. Adaptation resumes completed observations within the same bound configuration.

The Phase A commands above generate the development results used in the paper.
Adaptation evaluates the frozen bank on the held-out evaluation split for each
assignment. Model runs are sequential within each command sequence.

### Generate the additional comparisons

The final tables also require development candidate evaluation, clean-input
adaptation, and the fixed denoising baseline:

```bash
python reproduce.py paper revision prepare --run-id additional \
  --dev-assignment configs/paper/assignments/dev328.json
for stage in fixed-dev clean denoise; do
  python reproduce.py paper revision infer --run-id additional --stage "$stage" --model raft --batch 2
  python reproduce.py paper revision infer --run-id additional --stage "$stage" --model acv --batch 4
done
python reproduce.py paper revision shared-fixed-best --run-id additional --output-tag shared-fixed
```

Then use [additional analyses](ADDITIONAL.md#candidate-budget-and-consensus) to
compute consensus scores from the newly saved candidate pools and assemble the
comparison report. Set `--evidence` to your new adaptation journals; omitting it
uses the bundled observations. The same frozen signals, candidate bank, and
metric definitions apply throughout.

### Analyze the saved predictions

- Phase A writes per-image task metrics and 50k-pixel analysis pools. Use
  `paper table2` to recompute pixel failure-ranking metrics.
- Adaptation writes per-candidate metrics, cross-candidate pools, and dense maps
  for the predeclared examples. Use `paper samepixel` and `paper qualitative`
  for candidate ordering and prediction/assignment visualizations.
- To build the unified final report from new results, supply the complete
  observation layout through `paper report --evidence` and the corresponding
  task journals/proxy summaries through `--context-evidence`.

See [paper analysis](PAPER.md#cpu-analysis) for cache fields, commands, and the
report input layout. `paper report` with no input arguments always reproduces
the included results; it does not discover new runs automatically.

## Challenge: CroCo-Stereo Spring-FT

M03 uses eight frozen candidates including Identity. For each official
condition, run every candidate and then apply Tile Global selection to their
saved disparity/risk caches. Both reference eyes are predicted.

```bash
# Example: the Identity candidate for clean inputs.
python reproduce.py challenge infer --preset croco-07-fp16 \
  --inputs "$ROCO_ROOT/outputs/official_data/inputs" \
  --condition clean --candidate identity --run-id m03-clean-identity \
  --batch 16 --precision-scope forward-only

# Example: the Identity candidate for one corruption.
python reproduce.py challenge infer --preset croco-07-fp16 \
  --inputs "$ROCO_ROOT/outputs/official_data/inputs" \
  --condition gaussian_noise --candidate identity --run-id m03-gaussian_noise-identity \
  --batch 16 --precision-scope whole-call
```

Repeat with the remaining candidate IDs in `configs/challenge/bank8.json` and
all official conditions. The explicit autocast scopes match the recorded M03
settings. Use separate run IDs for each condition/candidate.

Follow [challenge reproduction](CHALLENGE.md) for the remaining steps:

**infer → index → select → layout → delta / package**

`index` checks the completed eight-candidate cohort; `select` saves Identity
and Tile Global predictions. `delta` measures clean-to-corrupted prediction
disagreement. `package` creates files using the official submission tool; it
does not upload them. Official clean accuracy requires evaluation by the
benchmark server. These steps use the new prediction arrays, whereas
`challenge report` displays the archived results.
