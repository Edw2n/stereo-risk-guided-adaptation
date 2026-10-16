# Challenge

The M03 workflow applies the risk-guided input-selection insights from the
RAFT-Stereo/ACVNet study to CroCo-Stereo Spring-FT.
See the [M03 benchmark result](https://spring-benchmark.org/531/) and
[RoCo-Spring leaderboard](https://roco-spring.github.io/evaluation.html).
Use the [command guide](COMMANDS.md) to distinguish archived reporting,
cache-based evaluation, inference, and packaging.

For environment installation and downloads, start with
[inference from scratch](INFERENCE.md#challenge-croco-stereo-spring-ft).
To display the recorded results without inference, use
[`challenge report`](#archived-results).

## CroCo inference

Inputs use `INPUTS/condition/test/sequence/frame_left|frame_right/*.png`.

```bash
# M02 baseline; both reference eyes, native resolution.
python reproduce.py challenge infer --preset croco-09-fp32 --inputs INPUTS --condition clean --run-id m02-clean --batch 16

# M03: run every frozen candidate for each condition, including clean.
python reproduce.py challenge infer --preset croco-07-fp16 --inputs INPUTS --condition clean --candidate identity --run-id m03-clean-identity --batch 16
```

Candidate IDs are in `configs/challenge/bank8.json`: **8 total, including Identity**. Repeat the second command with each ID and condition; commands are resumable and never schedule other conditions automatically. Batch counts reference images, so B16 processes eight stereo pairs with both eyes. The final short group is padded and duplicate outputs discarded. Use one process per GPU; check free VRAM before increasing batch.

M03 saves paired float32 disparity and risk arrays in lossless HDF5 under `<run>/cache/condition/sequence/frame_candidate.h5`. Selectors can then be changed/reaggregated on CPU without model inference. Geometry, transform order and risk definitions are frozen.

## Global selection

Index the eight completed candidate runs for one condition:

```bash
python reproduce.py challenge index --condition clean --run-id index-clean --runs   "$ROCO_ROOT/outputs/m03-clean-identity"   "$ROCO_ROOT/outputs/m03-clean-joint_03"   "$ROCO_ROOT/outputs/m03-clean-joint_11"   "$ROCO_ROOT/outputs/m03-clean-joint_14"   "$ROCO_ROOT/outputs/m03-clean-joint_15"   "$ROCO_ROOT/outputs/m03-clean-joint_22"   "$ROCO_ROOT/outputs/m03-clean-joint_23"   "$ROCO_ROOT/outputs/m03-clean-joint_28"
```

Use the resulting directory as `INDEX_DIRECTORY`. A manually prepared
`CACHE_INDEX.json` may instead map each candidate ID to its run directory.
All eight entries must describe the same condition and pair cohort.

```bash
python reproduce.py challenge select --cache INDEX_DIRECTORY --condition clean --run-id selected-clean --methods tile-global
```

Outputs are `<run>/predictions/Identity/` and `<run>/predictions/tile_disparity_std_unweighted_global/`, each with official condition/test layout. `--methods global` adds Head+LR+Census Global; `--methods all` also writes pixel diagnostics. M03 uses **Tile Global**.

Clean and corrupt inputs use the same selector, without GT or a clean reference. The clean reference is read only for subsequent ΔAbs evaluation.

## ΔAbs and packaging

Collect completed condition directories for **one method** in an official layout:

```bash
python reproduce.py challenge layout --run-id m03-layout   --method tile_disparity_std_unweighted_global --sources   "$ROCO_ROOT/outputs/selected-clean/predictions/tile_disparity_std_unweighted_global"   "$ROCO_ROOT/outputs/selected-brightness/predictions/tile_disparity_std_unweighted_global"
```

Pass all required condition directories in one invocation; the example lists
only two. This creates symlinks, not duplicate prediction arrays. Use the
printed path as `PREDICTIONS`. A clean package needs 2,000 maps; robust packaging
requires all 21 conditions, totaling 42,000 maps.

```bash
python reproduce.py challenge delta --clean PREDICTIONS --corrupt PREDICTIONS --run-id delta --clean-abs 0.474
python reproduce.py challenge package --layout PREDICTIONS --kind clean --tool /path/disp1_subsampling --run-id package-clean
python reproduce.py challenge package --layout PREDICTIONS --kind robust --tool /path/disp1_robust_subsampling --run-id package-robust
```

The local profile supplies the official binaries' library paths through `official_tool_library_paths`; edit this list on a fresh machine; see `provenance/official_tools.json`. Packaging checks the executable hash and expected file inventory. It **does not upload**. Obtain these tools from the official Spring/RoCo project resources; they are not bundled.

Native ΔAbs is mean absolute clean–corrupt prediction disagreement, not corrupted GT error. The full-native estimate differs from the server's subsampled evaluation. With an externally supplied clean Abs E0, the local estimate is `0.5*E0/3.4545 + 0.5*(E0+Δ)/18.4505`.

## Archived results

```bash
python reproduce.py challenge report --run-id challenge-results
```

This produces separate tables for archived official scores, server condition
breakdowns and local native disagreement. Local native disagreement uses both
reference eyes and an equal macro over the 20 corruptions. Official values are
archived observations, not new evaluation or live leaderboard claims. There
is no official overlap-0.7 Identity clean result, so its official RbS is not
estimated by substituting M03's selected clean score.

## Recorded inference settings

- M02 / entry 530: overlap 0.9; clean and 18 corruption conditions FP32; spatter and speckle_noise FP16. To reproduce that mixed campaign, use `croco-09-fp16` only for those two conditions. Do not label it uniform FP32.
- M03 / entry 531: overlap 0.7; clean inference used forward-only AMP; corruption inference used whole-call autocast. Set `--precision-scope whole-call` for those conditions. This difference is preserved in receipts; cross-GPU bitwise identity is not claimed.
- M01: FoundationStereo baseline is available through `--preset foundation`, B4, 32 iterations, hierarchical FP16 inference. It requires the pinned model/config pair and DINOv2 dependency.

The presets reproduce recorded experiments. Official scores are archived server results.

## Existing legacy M03 caches

`challenge select` also accepts the original read-only layout `cache/CONDITION/CANDIDATE/SEQUENCE/FRAME.h5`. Each `CACHE_INDEX.json` entry then has:

```json
{
  "identity": {
    "root": "/storage-volume/original-run",
    "layout": "legacy_v1",
    "evidence": {"path": "legacy_evidence.json", "sha256": "SHA256_OF_EVIDENCE_FILE"}
  }
}
```

Include all eight canonical IDs, with their own root/evidence if needed. Relative paths resolve against the index directory. Plain string entries retain the new `native_v1` layout.

The evidence file must use `schema: legacy_cache_attestation_v1` and contain:
- `cache_match`: exact `config_sha256`, `checkpoint_sha256`, `condition` from the actual cache (also bind `attempt` when available).
- `inference`: `checkpoint_sha256`, `upstream_commit`, `overlap`, `precision`, `precision_scope`, `GPU`, `batch` taken from the matching execution records.
- `sources`: file paths and SHA256 values for those execution records; relative paths resolve against the evidence file.
- `rationale`: which record/code establishes each missing field, especially the AMP scope.

This is an explicit audit attestation, not automatic discovery of missing settings. Do not fill unknown values by assuming the new defaults. Files, cache bindings and any fields already present are checked; contradictions are rejected. Original HDF5 metadata is never rewritten. Output metadata records the supplemented fields, attestation and source hashes. The adapter never rewrites the original cache or fills unknown execution settings from defaults.

## Execution provenance and resume

Selected DSP5 maps and selection journals carry candidate inference profiles (checkpoint, upstream commit, overlap, precision, AMP scope, GPU and batch). ΔAbs summaries/DOMAIN.json and PACKAGE_READY.json include condition-wise profiles, unknown-field flags and separate mixed-dtype/mixed-AMP-scope flags. `mixed_precision` is true for either difference; GPU/batch variation is reported separately. Unknown legacy prediction metadata stays explicitly unknown. Packaging reads metadata during inventory checks; it does not reread every prediction array to build this summary.

Resume preserves the historical contracts: completed 0.7 caches are reusable by execution/config fingerprint independent of the current GPU; 0.9 existing DSP5 maps also require matching GPU metadata. A different GPU for a partial 0.9 output needs a separate run ID. Precision/overlap changes always require a new run ID. This asymmetry is recorded, not a cross-device equivalence claim.
