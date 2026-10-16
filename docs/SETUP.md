# Setup

## Reference environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA H100 NVL, 95,830 MiB (93.6 GiB); one inference process |
| CPU | 2 × Intel Xeon Gold 6526Y; 32 physical cores / 64 logical CPUs |
| RAM | 503.6 GiB visible to Linux |
| Storage | `/storage1`, approximately 16.3 TiB total capacity |
| OS | Ubuntu 22.04.3 LTS, Linux 5.15.0-153-generic |
| Driver | NVIDIA 580.82.07 |
| Paper runtime | Python 3.13.13; PyTorch 2.11.0+cu128; NumPy 2.2.6 |
| CroCo runtime | Python 3.10.12; PyTorch 2.4.1 / CUDA 12.1; NumPy 1.26.4 |

The GPU model, capacity and driver were checked on 2026-10-16. The complete
runtime and hardware inventories are in `provenance/`. These describe the
reference environment, not minimum requirements. Available memory varies.

## Configure paths

Choose a large storage volume and run:

```bash
export ROCO_ROOT=/storage1/$USER/stereo-risk
python reproduce.py configure --storage-root "$ROCO_ROOT"
```

This creates `configs/local.json` and links the paper asset directories. It does
not download data or install software. Existing configurations are preserved;
edit the JSON to use existing environments, datasets or checkpoints.

## Install runtimes

With Python 3.13 and 3.10 available:

```bash
python3.13 -m venv "$ROCO_ROOT/envs/paper"
"$ROCO_ROOT/envs/paper/bin/python" -m pip install -r requirements/paper.txt
python3.10 -m venv "$ROCO_ROOT/envs/croco"
"$ROCO_ROOT/envs/croco/bin/python" -m pip install -r requirements/croco.txt
python reproduce.py doctor
python reproduce.py test
```

`g++` with OpenMP is required for the CPU rank/concordance helpers. CroCo uses
the upstream PyTorch RoPE implementation; no custom CUDA extension is required.
Use the headless OpenCV wheel in the requirements and do not install both
OpenCV distributions in the same environment.

Commands select the configured runtime automatically. Reporting does not need
a working GPU. The optional FoundationStereo baseline has its own runtime in
`requirements/foundation.txt`; it is not needed for the paper or M03.

## Inference settings

Paper defaults are FP32, TF32 disabled, native 1920 × 1080, RAFT batch 2 and ACV
batch 4. CroCo M03 uses FP16 autocast, overlap 0.7 and reference-image batch 16
(eight stereo pairs). The model forward, accumulation and AMP scope are recorded
separately. Do not change resolution, iterations, bank or precision implicitly.
Use a separate run ID for a batch override or other execution change.

For data and checkpoint sources, continue with [assets](ASSETS.md).
