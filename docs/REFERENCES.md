# References and software

## Models, data, and evaluation

| Work | Use in this repository | Reference and implementation |
|---|---|---|
| RAFT-Stereo — Lipson, Teed, and Deng, 3DV 2021 | Recurrent stereo backbone and late-update risk signals | [Paper](https://arxiv.org/abs/2109.07547) · [Code](https://github.com/princeton-vl/RAFT-Stereo) |
| Attention Concatenation Volume — Xu et al., CVPR 2022 | ACVNet backbone and attention/matching ambiguity signals | [Paper](https://arxiv.org/abs/2203.02146) · [Code](https://github.com/gangweiX/ACVNet) |
| CroCo v2 — Weinzaepfel et al., ICCV 2023 | Spring-finetuned CroCo-Stereo, overlapping-tile inference, and challenge extension | [Paper](https://arxiv.org/abs/2211.10408) · [Code](https://github.com/naver/croco) |
| FoundationStereo — Wen et al., CVPR 2025 | Optional screening baseline | [Project and paper](https://nvlabs.github.io/FoundationStereo/) · [Code](https://github.com/NVlabs/FoundationStereo) |
| DINOv2 — Oquab et al. | Frozen visual-feature dependency of the FoundationStereo baseline | [Paper](https://arxiv.org/abs/2304.07193) · [Code](https://github.com/facebookresearch/dinov2) |
| Spring — Mehl et al., CVPR 2023 | Public training data for the controlled study and official test evaluation | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mehl_Spring_A_High-Resolution_High-Detail_Dataset_and_Benchmark_for_Scene_Flow_CVPR_2023_paper.html) · [Benchmark](https://spring-benchmark.org/) |
| RobustSpring — Oei et al., ICLR 2026 | Official corruption data and clean-to-corrupted prediction disagreement | [Paper](https://openreview.net/forum?id=RebPBMrMmk) · [Dataset](https://doi.org/10.18419/DARUS-5047) |
| RoCo-Spring | Challenge evaluation, leaderboard, and submission conventions | [Website](https://roco-spring.github.io/) · [Evaluation](https://roco-spring.github.io/evaluation.html) · [Devkit](https://github.com/hmorimitsu/roco-spring-devkit) |

Upstream CroCo also credits [MAE](https://github.com/facebookresearch/mae),
[timm](https://github.com/huggingface/pytorch-image-models), and
[UniMatch](https://github.com/autonomousvision/unimatch). Their existing notices
remain applicable to the corresponding upstream components.

The paper bibliography is in [references.bib](../references.bib).

## Software dependencies

| Package | Use | Reference |
|---|---|---|
| PyTorch / torchvision | Model execution, tensor operations, and upstream vision utilities | [PyTorch](https://pytorch.org/) · [torchvision](https://github.com/pytorch/vision) |
| NumPy | Array operations and numerical aggregation | [NumPy](https://numpy.org/) · [Paper](https://doi.org/10.1038/s41586-020-2649-2) |
| SciPy | Rank statistics and Latin hypercube candidate-bank construction | [SciPy](https://scipy.org/) · [LatinHypercube](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.qmc.LatinHypercube.html) |
| scikit-learn | AUROC and average precision | [Project](https://scikit-learn.org/) · [Paper](https://jmlr.org/papers/v12/pedregosa11a.html) |
| h5py / HDF5 | Chunked prediction and analysis-pool storage | [h5py](https://www.h5py.org/) · [HDF5](https://www.hdfgroup.org/solutions/hdf5/) |
| OpenCV / Pillow | Image processing, transformations, and image I/O | [OpenCV](https://opencv.org/) · [Pillow](https://python-pillow.github.io/) |
| Matplotlib | Scientific plots and qualitative figures | [Project](https://matplotlib.org/) · [Paper](https://doi.org/10.1109/MCSE.2007.55) |
| einops / timm | Tensor rearrangement and upstream model components | [einops](https://github.com/arogozhnikov/einops) · [timm](https://github.com/huggingface/pytorch-image-models) |
| Requests / gdown | Explicit downloads from the recorded asset sources | [Requests](https://requests.readthedocs.io/) · [gdown](https://github.com/wkentaro/gdown) |
| tqdm | Progress reporting | [Project](https://github.com/tqdm/tqdm) |

Exact versions and additional runtime dependencies are listed in
[paper requirements](../requirements/paper.txt),
[CroCo requirements](../requirements/croco.txt), and
[optional FoundationStereo requirements](../requirements/foundation.txt).
See [third-party notices](../THIRD_PARTY.md) for model and vendored-code licenses.
