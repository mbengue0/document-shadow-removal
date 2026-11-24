# CS-4212 project progress: document shadow removal with text preservation

Student: Mouhamadou Mbengue. Last updated: 24 November 2026.
Status: revised proposal presented on 24 November 2026; the teacher accepted the direction.

## Project in a few sentences
A phone photo of a document often has a shadow on it. We train a U-Net to take a shadowed
document crop and predict the clean, shadow-free crop. Real paired photos from SD7K are the
training data. The experiment asks whether adding an edge-preservation term to the usual L1
pixel loss keeps text sharper, compared against L1 alone and one classical correction baseline.

## Teacher decisions (24 November 2026, given verbally at the proposal presentation)
1. Dataset size: about 64,800 paired 256x256 crops from SD7K photos of about 350 pages is accepted
   for the "more than 50,000 rows" requirement.
2. SD7K is accepted even though it is not African data.
3. Data delivery: share a Google Drive link to the downloaded data (instead of putting images in
   `dataset.csv`).
4. New request: add document photos taken with my own phone. The teacher wants to see them in the
   final presentation.

Still to confirm later: the assignment number for the submission file names.

## Phone photo plan (proposed)
- Role: a separate held-out test set, never used for training or tuning. It tests whether a model
  trained on SD7K (fixed Canon camera) works on a different camera and on local documents.
- Paired set: phone fixed above the page, focus and exposure locked, HDR/night mode off, no flash.
  One shot with a shadow, then remove the object and shoot again without moving the phone.
  Saved as `data/phone/input/` and `data/phone/target/` with matching file names, so the same
  manifest script checks alignment.
- Unpaired set: ordinary handheld shadowed photos, with no clean reference, shown as visual
  examples only (no PSNR/SSIM possible).
- No documents with personal information, since the data will be shared by link.
- Results on phone photos are reported separately from SD7K results.

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
- [x] Stage 1e: Revised proposal presented; teacher decisions recorded above.
- [ ] Stage 1d: Run the manifest on a real SD7K sample (waiting for the download).
- [ ] Stage 1f: Capture phone photos (paired and unpaired) and run the manifest on the paired set.
- [ ] Stage 2: Inspect the real sample visually: check alignment, check grouping accuracy, decide between the full set and a text-only subset.
- [ ] Stage 3: Document-level split (before cropping), plus a crop manifest. Count documents, pairs and patches per split.
- [ ] Stage 4: PyTorch dataset + compact U-Net + L1/Sobel-edge loss. Forward/backward check and tiny-batch overfit.
- [ ] Stage 5: Pilot epoch on the GPU: time, memory, cost estimate for all runs.
- [ ] Stage 6: Small validation pilot to fix the edge weights. Then full runs.
- [ ] Stage 7: Tiled full-page inference, evaluation on SD7K test pages and phone photos, report, final presentation.

## Results
None yet. No training has been run.
