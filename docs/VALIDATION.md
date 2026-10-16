# Validation

Validation uses the pinned reference environments and archived experiment
outputs. Results are recorded in [local_release.json](../validation/local_release.json).

| Check | Result |
|---|---|
| CPU metric, selector, sampling, and workflow tests | 36 passed |
| Final paper comparisons | 5,111 numeric values match exactly |
| Three-assignment reports | Four JSON summaries match exactly |
| Candidate-budget and condition reports | All compared values match exactly |
| M03 selection from one saved native stereo pair | Both-eye outputs match bitwise |
| Numerical kernels and DINOv2 source | 35 kernel files and 203 vendor files unchanged |

Earlier checks include one-pair GPU smoke tests for RAFT-Stereo, ACVNet, and
CroCo, plus cached Table 2, candidate-ordering, and qualitative analyses.
Fresh environment installation and full-dataset inference were not repeated
for this release. Numerical identity across GPUs is not assumed.

Packaging tests use synthetic inputs and a stub executable. Actual submission
files require the separately downloaded, hash-verified official tools and the
complete prediction inventory.

Verify the distributed files with:

```bash
python scripts/verify_release.py
```
