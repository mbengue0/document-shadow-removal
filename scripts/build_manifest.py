"""Stage 1: build a pair manifest for SD7K (or any input/ + target/ paired folder) and inspect it.

What it does, in plain terms:
  1. Finds every folder called `input` that has a sibling folder called `target`
     (this is the layout the official SD7K loader expects: <split>/input, <split>/target).
  2. Pairs files by file name stem. Unmatched files are listed, never guessed.
  3. For each pair it records paths, the original split, image sizes, file checksums,
     and two cheap quality checks:
       - global_shift_px: estimated translation between input and target
         (phase correlation on edge maps of downscaled grayscale images).
         A well-aligned pair should be close to (0, 0).
       - mean_abs_diff: average pixel difference on the downscaled grayscale images.
  4. Computes a 64-bit difference hash of each CLEAN target, then groups targets whose
     hashes are close. Same printed page photographed under different shadows should
     give near-identical clean targets, so these groups are CANDIDATE source-document IDs.
     They are a heuristic, not ground truth: check them visually before splitting on them.

Nothing here modifies the images. Output: manifest.csv, unmatched.csv, summary.json.

Usage:
  python build_manifest.py --root /path/to/SD7K --out ../manifests [--limit 200] [--hash-threshold 6]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SMALL = 512  # long side of the downscaled copy used for checks


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_small_gray(path):
    with Image.open(path) as im:
        size = im.size  # (width, height) of the original
        im = im.convert("L")
        im.thumbnail((SMALL, SMALL), Image.BILINEAR)
        return size, np.asarray(im, dtype=np.float32) / 255.0


def edge_map(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def phase_shift(a, b):
    """Return (dy, dx) shift of b relative to a, in pixels of the small images."""
    if a.shape != b.shape:
        return None
    ea, eb = edge_map(a), edge_map(b)
    ea -= ea.mean()
    eb -= eb.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(ea * win), np.fft.fft2(eb * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    corr = np.fft.ifft2(r).real
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx)


def dhash(gray_small, size=8):
    im = Image.fromarray((gray_small * 255).astype(np.uint8)).resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(im, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def find_pair_dirs(root):
    for inp in sorted(root.rglob("input")):
        tgt = inp.parent / "target"
        if inp.is_dir() and tgt.is_dir():
            yield inp.parent.relative_to(root).as_posix() or ".", inp, tgt


def images_by_stem(folder):
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXT}


def group_hashes(rows, threshold):
    """Union-find over targets whose dHash differs by <= threshold bits."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [r["target_dhash"] for r in rows]
    for i in range(len(rows)):  # O(n^2): fine for ~7.6k pairs (~29M cheap comparisons)
        for j in range(i + 1, len(rows)):
            if hamming(hashes[i], hashes[j]) <= threshold:
                parent[find(i)] = find(j)
    roots = {}
    for i, r in enumerate(rows):
        r["candidate_group"] = roots.setdefault(find(i), f"g{len(roots):05d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0, help="max pairs per split (0 = all)")
    ap.add_argument("--hash-threshold", type=int, default=6)
    ap.add_argument("--shift-tolerance", type=int, default=2, help="pixels at the downscaled size")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, unmatched = [], []
    splits = list(find_pair_dirs(args.root))
    if not splits:
        sys.exit(f"No <split>/input + <split>/target folders found under {args.root}")

    for split, inp_dir, tgt_dir in splits:
        inp, tgt = images_by_stem(inp_dir), images_by_stem(tgt_dir)
        for stem in sorted(set(inp) ^ set(tgt)):
            side = "input_only" if stem in inp else "target_only"
            unmatched.append({"split": split, "stem": stem, "side": side,
                              "path": str(inp.get(stem) or tgt.get(stem))})
        stems = sorted(set(inp) & set(tgt))
        if args.limit:
            stems = stems[: args.limit]
        for k, stem in enumerate(stems, 1):
            row = {"pair_id": f"{split}/{stem}", "split": split,
                   "input_path": str(inp[stem]), "target_path": str(tgt[stem])}
            try:
                (iw, ih), a = load_small_gray(inp[stem])
                (tw, th), b = load_small_gray(tgt[stem])
            except Exception as e:  # corrupt or undecodable file
                row.update(status="decode_error", note=repr(e)[:200])
                rows.append(row)
                continue
            shift = phase_shift(a, b)
            row.update(
                input_w=iw, input_h=ih, target_w=tw, target_h=th,
                same_size=(iw, ih) == (tw, th),
                input_sha256=sha256(inp[stem]), target_sha256=sha256(tgt[stem]),
                input_bytes=inp[stem].stat().st_size, target_bytes=tgt[stem].stat().st_size,
                mean_abs_diff=round(float(np.abs(a - b).mean()), 4) if a.shape == b.shape else "",
                shift_dy=shift[0] if shift else "", shift_dx=shift[1] if shift else "",
                target_dhash=dhash(b),
            )
            flags = []
            if not row["same_size"]:
                flags.append("size_mismatch")
            if shift and max(abs(shift[0]), abs(shift[1])) > args.shift_tolerance:
                flags.append("possible_misalignment")
            if row["input_sha256"] == row["target_sha256"]:
                flags.append("identical_files")
            row["status"] = "ok" if not flags else "flag:" + "+".join(flags)
            rows.append(row)
            if k % 200 == 0:
                print(f"  {split}: {k}/{len(stems)}", flush=True)

    good = [r for r in rows if "target_dhash" in r]
    group_hashes(good, args.hash_threshold)

    # exact duplicate clean targets across different pairs are a strong same-page signal
    dup = Counter(r["target_sha256"] for r in good)
    for r in good:
        r["target_exact_dup_count"] = dup[r["target_sha256"]]
        r["target_dhash"] = f"{r['target_dhash']:016x}"

    cols = ["pair_id", "split", "input_path", "target_path", "status", "note",
            "input_w", "input_h", "target_w", "target_h", "same_size",
            "shift_dy", "shift_dx", "mean_abs_diff", "input_sha256", "target_sha256",
            "input_bytes", "target_bytes", "target_dhash", "target_exact_dup_count", "candidate_group"]
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "unmatched.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "stem", "side", "path"])
        w.writeheader()
        w.writerows(unmatched)

    by_split = defaultdict(lambda: {"pairs": 0, "ok": 0, "flagged": 0, "groups": set()})
    for r in rows:
        s = by_split[r["split"]]
        s["pairs"] += 1
        s["ok" if r.get("status") == "ok" else "flagged"] += 1
        if "candidate_group" in r:
            s["groups"].add(r["candidate_group"])
    groups_split = defaultdict(set)
    for r in good:
        groups_split[r["candidate_group"]].add(r["split"])
    summary = {
        "root": str(args.root), "limit_per_split": args.limit,
        "hash_threshold": args.hash_threshold, "shift_tolerance_small_px": args.shift_tolerance,
        "splits": {k: {"pairs": v["pairs"], "ok": v["ok"], "flagged": v["flagged"],
                       "candidate_groups": len(v["groups"])} for k, v in by_split.items()},
        "unmatched_files": len(unmatched),
        "status_counts": dict(Counter(r.get("status") for r in rows)),
        "image_sizes": dict(Counter(f"{r['input_w']}x{r['input_h']}" for r in good).most_common(10)),
        "candidate_groups_total": len({r["candidate_group"] for r in good}),
        "groups_spanning_multiple_splits": sum(1 for v in groups_split.values() if len(v) > 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
