# Paper experiments

For new predictions, start with [inference from scratch](INFERENCE.md#paper-raft-stereo-and-acvnet).
This page describes analysis of completed runs.

## CPU analysis

The checkout includes scalar evidence and summarized proxy CSVs, **not** the analysis-pool HDF5 or dense NPZ arrays. `paper report` recomputes Table 1/3 from journals; proxy-level and condition-response plots redraw the saved CSV values. The commands below actually recalculate Table 2 and same-pixel metrics only when their cache inputs are supplied.

| Command | Required cache/data references in the journal |
|---|---|
| `table2` | `analysis_pool.shard`, `.key`, `.coordinates`, `.coordinate_key`; 50k scores/errors/D1 and common coordinates |
| `samepixel` | `pool.path`, `.group`; 31 candidate risks/errors/D1, validity mask |
| `qualitative` | `dense_qualitative_path`; native predictions/assignments, plus downloaded Spring RGB/GT |

These are produced by `paper phase-a` or `paper adaptation` respectively. If reusing caches on a different host, make a copy of the scalar journal and update its path fields to the transferred immutable files; never alter the source journal. Supplying only the summary CSV cannot recover these inputs. No analysis command starts inference to fill missing data.

`samepixel` builds its C++ concordance helper with `g++` on first use under the new output directory (`cache/native_helpers`). This is CPU compilation, not GPU inference.

```bash
python reproduce.py paper report --run-id paper
python reproduce.py paper table2 --journal /storage1/.../raft/samples.jsonl --pixels 20000 --run-id table2-20k
python reproduce.py paper table2 --journal /storage1/.../raft/samples.jsonl --pixels 40000 --run-id table2-40k
python reproduce.py paper table2 --journal /storage1/.../raft/samples.jsonl --common-support --run-id table2-secondary
python reproduce.py paper samepixel --journal /storage1/.../raft/observations.jsonl --run-id ordering
python reproduce.py paper qualitative --journal /storage1/.../raft/observations.jsonl --pair 0014/0154 --run-id qualitative-preview
```

`paper report` regenerates the final adaptation table with shared Fixed-best,
consensus and denoising comparisons, clean-input metrics, development task
metrics, and plot previews. It reports means and standard deviations over three
assignments alongside paired sequence-cluster confidence intervals. Tables are
saved as CSV/LaTeX and previews as PNG/PDF. Proxy plots redraw saved summaries. `table2` recomputes AUROC/AP/Spearman from analysis pools, without inference. `samepixel` reads the 31-candidate pool; it is a distinct candidate-axis diagnostic.

The default inputs are `evidence/camera_ready` for final comparisons and
`evidence/paper` for development task journals and proxy summaries. For new
completed runs, supply `--evidence DIRECTORY` with the same final-evidence
layout: `adaptation/seed{101,202,303}/{raft,acv}.jsonl`,
`fixed-dev/{raft,acv}.jsonl`, `clean/{raft,acv}.jsonl`,
`denoise/{raft,acv}.jsonl`, `consensus.jsonl`, and `clean_choices.csv`.
Use `--context-evidence DIRECTORY` for development journals (`phase_a`) and
proxy summary directories matching the bundled layout. Missing inputs are not
replaced by inference. Use a new run ID for the unified report.

## Definitions

- Native error: minimum absolute error over the four Spring GT subpixels, in native disparity units. D1 requires error >3 px **and** relative error >5%; preserve GT dtype for the threshold.
- Table 1/3: pool valid-pixel sufficient statistics within condition, then equal-weight six-corruption macro. Clean is separate.
- Table 2: condition-pooled sampled pixels, then corruption macro; sequence-cluster bootstrap. GT<192 is secondary, with the same saved coordinates.
- Frozen signals: RAFT `late_std(last_n=4, iterations=32)`; ACV `attention_entropy`.
- Signal-Global: lowest mean within-pixel candidate rank over geometry x≥4. Equal pixel ranks; deterministic seeded global ties.
- Signal-Pixel: fuse candidate disparities by minimum risk; outside support use Identity. No RGB mosaic inference.
- Global Oracle: image D1, then Abs, then 1px. Pixel Oracle: native absolute-error argmin, evaluation only.

Three-seed results average per-seed macro metrics. Recovery averages per-seed ratios. SD measures assignment variation; 95% CIs resample whole sequences with common weights across methods/seeds. Candidate ordering and qualitative examples remain assignment-101 diagnostics.

`qualitative` requires a predeclared example with saved dense maps. It writes global/pixel PNG and PDF, exact task metrics, candidate-color fractions and provenance; it never creates new predictions. A pair without dense maps is rejected. 

## Final comparisons

`paper report` recomputes one shared development Fixed-best (`joint_14`),
the final adaptation table, denoising and consensus comparisons, paired
Signal-Global differences, and clean-preservation metrics from the included
scalar observations. It does not select a different signal or bank.

Corrupted-input consensus choices are recomputed from saved candidate scores.
The clean consensus metrics use saved candidate choices; extracting those
scores again requires the disparity pools. The output records this distinction.
See [additional analyses](ADDITIONAL.md) for rerunning their inference and
cache-backed analysis.
