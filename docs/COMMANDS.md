# Command guide

Run commands from the repository root after [setup](SETUP.md).
`python reproduce.py --help` lists workflow groups; append `--help` to a group
or command to see its options. A `--run-id` identifies a distinct output
directory; use a new ID when changing the configuration or input evidence.

For the included results, start with [CPU reporting](../README.md#reproduce-reported-results).
For new predictions, follow [inference from scratch](INFERENCE.md).

## Setup and checks

| Command | Purpose | What it changes |
|---|---|---|
| `configure --storage-root PATH` | Choose where environments, datasets, weights, and outputs live. | Creates the local path configuration and asset links. |
| `doctor` | Display the configured paper Python runtime, key installed package versions, and paths. | Prints information only; not a complete asset or GPU validation. |
| `test` | Check metric definitions, selectors, sampling, cache compatibility, and command behavior. | Runs CPU tests with synthetic or small local inputs. |
| `assets list` | Show the pinned models, upstream revisions, and checkpoint sources. | Prints information only. |
| `assets verify-vendor` | Verify the bundled DINOv2 file inventory and hashes. | Reads local files only. |
| `assets clone-model`, `download-model`, `download-paper`, `download-official` | Obtain explicitly requested upstream code, weights, or data. | Downloads to the configured paths. See [assets](ASSETS.md). |

`doctor` prints environment information; model smoke tests use the inference commands.

## Paper results

| Command after `paper` | Inputs | Outputs | GPU inference |
|---|---|---|---|
| `report` | Included final scalar observations and proxy summary CSVs | Final task/adaptation tables, shared Fixed-best, consensus, denoising, clean behavior, paired differences, and plot previews | No |
| `table2` | Saved 50k analysis-pool caches and journal | AUROC, AP, and Spearman reaggregated at the requested sample size | No |
| `samepixel` | Saved cross-candidate pools and journal | Candidate-axis ordering diagnostics | No |
| `qualitative` | Saved dense examples, journal, RGB, and GT | Prediction/error/assignment figure previews and metrics | No |
| `phase-a` | Dataset, pretrained model, and fixed split/config | Fresh task predictions, scalar observations, and analysis pools | Yes |
| `adaptation` | Dataset, pretrained model, frozen signal/bank, and assignment | Fresh candidate predictions, selection results, and caches | Yes |

`report` is the single entry point for the final paper results. The former
`camera-ready` command remains an alias for compatibility; it produces the same
outputs. `table2` and `samepixel` recompute pixel-level metrics from saved arrays
rather than redrawing summary CSVs.
The `paper revision` group contains the additional experiment workflows;
see [additional analyses](ADDITIONAL.md).

## Challenge results and submission files

| Command after `challenge` | Purpose | GPU inference |
|---|---|---|
| `report` | Render archived server scores, per-corruption scores, and saved native disagreement summaries | No |
| `infer` | Generate both-eye predictions and, for M03, candidate risk caches | Yes |
| `index` | Create an index of the eight completed candidate runs for one condition | No |
| `select` | Apply the frozen risk selector to candidate caches | No |
| `layout` | Assemble one method's completed conditions in the official directory layout using links | No |
| `delta` | Recalculate clean-to-corrupted prediction disagreement from prediction arrays | No |
| `package` | Use the hash-pinned official tool to create clean or robust submission files | No |

`challenge report` reads archived results. `challenge delta` calculates a new
local disagreement summary from saved predictions. Official clean GT scores
require evaluation by the benchmark server; the test GT is not bundled.
`challenge package` creates an upload file but does not upload it.

The M03 workflow is **infer → index → select → layout → delta/package**.
See [challenge reproduction](CHALLENGE.md) for complete examples and recorded
precision settings. Cached analyses never start inference to fill missing files.
