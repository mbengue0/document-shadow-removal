# Document Shadow Removal with Deep Learning and Text Preservation

CS-4212 Deep Learning and Neural Networks project by Mouhamadou Mbengue.

A U-Net is trained to turn a shadowed crop of a document photograph into its clean, shadow-free
version, using real paired photographs from the SD7K dataset. The experiment asks:

> Does adding an edge-preservation term to a reconstruction loss improve document shadow removal
> while preserving text details?

Status: early stage. See [PROGRESS.md](PROGRESS.md) for the checklist, verified facts and open
questions. No model has been trained yet and no results exist.

## Data

SD7K: Li, Z., Chen, X., Pun, C.-M., and Cun, X. (2023). *High-Resolution Document Shadow Removal via
A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net.* ICCV.
Download links are in the official repository: https://github.com/CXH-Research/DocShadow-SD7K

Images are not stored in this repository. Place the dataset under `data/SD7K/` with the official
`<split>/input/` and `<split>/target/` folders.

## Stage 1: build the pair manifest

```bash
pip install -r requirements.txt
python scripts/build_manifest.py --root data/SD7K --out manifests --limit 50   # small sample first
```

This writes `manifests/manifest.csv`, `unmatched.csv` and `summary.json`: one row per image pair,
with sizes, checksums, an alignment check and a candidate source-document group.

## References

Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for Biomedical
Image Segmentation. https://arxiv.org/abs/1505.04597

## AI assistance disclosure

In line with the course AI usage policy, an AI assistant (Claude, by Anthropic) was used to help
verify dataset sources, draft the data inspection script and organize project notes. Design choices,
review of the code and all reported results are the author's responsibility. This section will be
updated as the project progresses, stating where and how AI was used.
