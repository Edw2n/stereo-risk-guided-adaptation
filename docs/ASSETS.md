# Installation and assets

All data, weights, environments and outputs belong on `storage1`. No training is required.

Create `configs/local.json` and the Python environments using
[setup](SETUP.md). All example paths are configurable. Existing files with
incorrect hashes are rejected rather than overwritten.

## Models

```bash
python reproduce.py assets list
python reproduce.py assets clone-model --model raft
python reproduce.py assets download-model --model raft
# repeat with acv and croco
```

| Model | Checkpoint | Source |
|---|---|---|
| RAFT-Stereo | `raftstereo-sceneflow.pth` | [official pretrained archive](https://github.com/princeton-vl/RAFT-Stereo) |
| ACVNet | `ACVNet/sceneflow.ckpt` | [official weights](https://github.com/gangweiX/ACVNet); Google Drive ID in `configs/assets.json` |
| CroCo-Stereo Spring-FT | `crocostereo_finetune_spring.pth` | [NAVER stereo checkpoints](https://github.com/naver/croco/blob/master/stereoflow/README.MD) |
| FoundationStereo | `23-51-11/model_best_bp2.pth` + matching `cfg.yaml` | [official repository](https://github.com/NVlabs/FoundationStereo) |

Every model has an exact upstream commit and weight SHA256 in `configs/assets.json`. Existing assets with mismatching hashes are rejected. RAFT/ACV use SceneFlow checkpoints; CroCo Spring-FT was trained on Spring train, so its local Spring-train scores are not unseen-data evidence.

For Foundation, download the upstream 23-51-11 model/config pair manually, set paths in `configs/local.json`, and preserve the YAML SHA in `configs/challenge/foundation.json`. Its separate environment is recorded in `requirements/`; the exact historical DINOv2 code is included under `vendor/dinov2` with upstream licenses. Use trusted official checkpoints only.

## Data

- [Spring](https://doi.org/10.18419/DARUS-3376): public train RGB/disparity for paper experiments; official test RGB for challenge.
- [RobustSpring](https://doi.org/10.18419/DARUS-5047): official corrupted test RGB.

```bash
# Fetch only the frozen 500-pair paper manifest, using ZIP byte ranges.
python reproduce.py assets download-paper
# Download/extract one official condition at a time.
python reproduce.py assets download-official --condition clean
python reproduce.py assets download-official --condition gaussian_noise
```

Paper assets retain ZIP member names, byte counts and CRCs in `configs/paper/split500.json`. No resampling is done. Official archive IDs/checksums are frozen in `provenance/official_data_manifest.json`; outputs go to `<storage_root>/official_data/inputs`. Official test GT is not downloaded or available locally. Respect upstream data/model licenses and access requirements.

For an offline server, put the verified files at these same paths. Model and data download commands are explicit; running reports does not download anything.

Foundation environment: `requirements/foundation.txt`. The 23-51-11 model is Google Drive ID `1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe`; its YAML is ID `1tidGICH1_kTUUqi42aboKscuMY4IK_Xr`. Set `foundation_checkpoint`, `foundation_cfg` and `foundation_upstream` in the local profile. Foundation loads the frozen local DINOv2 snapshot, whose tree hash matches the recorded source snapshot; it does not fetch a floating `main`.

Official submission tools: [download archive](https://cloud.visus.uni-stuttgart.de/index.php/s/RSgYPXdKTyAFfWy/download). Verify the archive and executable SHA256 values in `provenance/official_tools.json` before use.

Verify the complete 203-file vendored source tree without a model or GPU:

```bash
python reproduce.py assets verify-vendor
```

This checks the distributed file inventory, individual SHA256 values and the recorded tree-hash algorithm. Source `data/` modules are included; no dataset is bundled.
