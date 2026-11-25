# Document Shadow Removal with Deep Learning and Text Preservation

CS-4212 Deep Learning and Neural Networks project by Mouhamadou Mbengue.

A U-Net is trained to turn a shadowed crop of a document photograph into its clean, shadow-free
version, using real paired photographs from the SD7K dataset. The experiment asks:

> Does adding an edge-preservation term to a reconstruction loss improve document shadow removal
> while preserving text details?

Status: data pipeline and training code written and checked on synthetic images. No model has
been trained on real data yet and no results exist. See [PROGRESS.md](PROGRESS.md).

## Data

SD7K: Li, Z., Chen, X., Pun, C.-M., and Cun, X. (2023). *High-Resolution Document Shadow Removal via
A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net.* ICCV.
Download links are in the official repository: https://github.com/CXH-Research/DocShadow-SD7K

Images are not stored in this repository. Place the dataset under `data/SD7K/` with the official
`<split>/input/` and `<split>/target/` folders.

## Pipeline

```bash
pip install -r requirements.txt

# 1. Check every pair: decoding, sizes, alignment, page groups
python scripts/build_manifest.py --root data/SD7K --out manifests --limit 50   # small sample first

# 2. Split by page group (no page in two splits)
python scripts/make_split.py --manifest manifests/manifest.csv --out manifests/split.csv

# 3. Fixed list of 256x256 training/validation crops (the dataset "rows")
python scripts/make_crops.py --split manifests/split.csv --out manifests/crops.csv

# 4. Sanity check, then the experimental conditions (only --lambda-edge changes)
python train.py --crops manifests/crops.csv --overfit-steps 300 --out runs/overfit
python train.py --crops manifests/crops.csv --lambda-edge 0   --out runs/l1_seed0
python train.py --crops manifests/crops.csv --lambda-edge 0.5 --out runs/edge0.5_seed0
```

Code check on synthetic images (not project data): `python tests/smoke_test.py`

| File | Purpose |
|---|---|
| `src/model.py` | Compact U-Net, RGB in and out, residual output starting as the identity |
| `src/losses.py` | L1 loss and Sobel edge loss on luminance |
| `src/data.py` | Serves the crops listed in `crops.csv` |
| `train.py` | Training, validation, logging (`log.csv`) and checkpoint selection by validation L1 |

## References

Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical
Image Segmentation. https://arxiv.org/abs/1505.04597

## AI assistance disclosure

In line with the course AI usage policy, an AI assistant (Claude, by Anthropic) was used to help
verify dataset sources, draft the data inspection script and organize project notes. Design choices,
review of the code and all reported results are the author's responsibility. This section will be
updated as the project progresses, stating where and how AI was used.
