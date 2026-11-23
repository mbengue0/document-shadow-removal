# CS-4212 project progress: document shadow removal with text preservation

Student: Mouhamadou Mbengue. Last updated: 23 November 2026.
Status: proposed direction. Teacher approval for this topic is **not yet confirmed**.

## Project in a few sentences
A phone photo of a document often has a shadow on it. We train a U-Net to take a shadowed
document crop and predict the clean, shadow-free crop. Real paired photos from SD7K are the
training data. The experiment asks whether adding an edge-preservation term to the usual L1
pixel loss keeps text sharper, compared against L1 alone and one classical correction baseline.

## Questions only the teacher can answer
1. Do more than 50,000 paired 256x256 patches from roughly 6,500 SD7K training photos of about
   350 source documents meet the "more than 50,000 rows" requirement?
2. Does the African data requirement apply to this DL project? SD7K is not African data.
3. Submission packaging for an image dataset: is a `dataset.csv` path/crop manifest acceptable
   in place of the images? Which assignment number goes in the file names?

## Verified facts (with sources)
| Fact | Source |
|---|---|
| 7,620 shadow/shadow-free pairs in total | ICCV 2023 paper (arXiv 2308.14221) |
| 6,479 train / 760 test reported. These add up to 7,239, not 7,620, and the paper does not explain why | same paper |
| More than 350 documents and over 30 occluder types | paper and repository README |
| Resolution 2462 x 3699; Canon EOS M6 on a fixed stand with fixed settings; occluder removed between shots; no registration step described | paper |
| Download: OneDrive (about 9 GB compressed, about 100 GB original) or OpenXLab | repository README |
| Repository archived on 30 Mar 2026 (read-only); code under MIT; no separate dataset licence stated | repository page |
| Official loader expects `<split>/input/` and `<split>/target/` folders, paired by sorted filename | `data/dataset_RGB.py` in the repo |
| Official training also adds **synthetic** shadows (albumentations RandomShadow) | same file. We will not copy this, because our plan uses real shadows only |
| No per-document ID is described anywhere | paper and repository. So grouping must be inferred |

## Environment (measured)
- This workspace: 2 CPUs, 7 GB RAM, **no GPU**, about 31 GB free disk. PyTorch 2.15 installed.
  It is fine for data inspection and small CPU tests. It is **not** suitable for real training.
- OneDrive and OpenXLab are blocked from this workspace. The dataset has to come from you.
- Still needed: your training hardware (GPU model, VRAM, time budget). Colab or a lab GPU?

## Checklist
- [x] Stage 0: Read the handoff and the old proposal deck (CIFAR-10 augmentation, 12 Nov). Deck kept for the record and not edited.
- [x] Stage 1a: Verify the SD7K source, counts, capture method, layout and terms (see table above).
- [x] Stage 1b: Write `scripts/build_manifest.py` (pairing, decoding, sizes, checksums, alignment check, candidate document groups).
- [x] Stage 1c: Test the script logic on synthetic files (code test only, not data). It flagged a shifted pair, a corrupt file, an unmatched file and the same page appearing in both train and test.
- [ ] Stage 1d: **Blocked:** run the manifest on a real SD7K sample (you download it).
- [ ] Stage 2: Inspect the real sample visually: check alignment, check grouping accuracy, decide between the full set and a text-only subset.
- [ ] Stage 3: Document-level split (before cropping), plus a crop manifest. Count documents, pairs and patches per split.
- [ ] Stage 4: PyTorch dataset + compact U-Net + L1/Sobel-edge loss. Forward/backward check and tiny-batch overfit.
- [ ] Stage 5: Pilot epoch on the GPU: time, memory, cost estimate for all runs.
- [ ] Stage 6: Small validation pilot to fix the edge weights. Then full runs (only after teacher answers).
- [ ] Stage 7: Tiled full-page inference, evaluation, report, presentation revision.

## Results
None yet. No training has been run.
