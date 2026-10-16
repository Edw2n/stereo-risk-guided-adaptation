# Licenses

Original experiment orchestration and analysis code is released under the
[MIT License](LICENSE). This grant does not replace third-party licenses.

| Component | License / notice | Distribution |
|---|---|---|
| RAFT-Stereo | [MIT](licenses/RAFT-Stereo.txt) | Download the pinned upstream code and weights. |
| ACVNet | [MIT](licenses/ACVNet.txt) | Download the pinned upstream code and weights. |
| CroCo | [CC BY-NC-SA 4.0 and upstream notices](licenses/CroCo.txt) | Download the pinned upstream code and checkpoint. CroCo-derived tiling and model adapters retain these terms. |
| FoundationStereo | [Upstream license](licenses/FoundationStereo.txt) | Optional baseline; download separately. |
| DINOv2 | [Apache 2.0](vendor/dinov2/LICENSE) | The pinned source tree is included unmodified, with its notices. |
| Spring / RobustSpring | Official dataset terms | Data and ground truth are not redistributed. |

CroCo-derived files are `kernels/challenge/screening/adapter_croco.py`,
`croco_proxy_extract.py`, `croco_proxy_extended_extract.py`, and
`croco_fp16_proxy.py`. These instrument upstream inference to expose risk;
their CroCo-derived portions are not relicensed under MIT. All model weights
retain their upstream terms. Exact revisions and checksums are in
`configs/assets.json` and `provenance/`.
