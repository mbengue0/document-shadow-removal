"""Stage 3b: reproducible list of training / validation crops.

One row = one training example: a square crop at the same (x, y) in the shadowed image and
its clean target. Crops are not saved as image files; positions are stored, and the training
code cuts them on the fly. This file is the project's "rows" table (dataset.csv).

Sampling rule (fixed before any training):
  - --per-image crops per pair, size --size, positions drawn uniformly at random with a
    seed derived from the pair ID, so the list is identical on every run.
  - Mostly blank crops teach little about text. A crop "has content" if the grey-level
    standard deviation of its CLEAN target is >= --min-std. At most --blank-frac of the
    crops of an image may be blank, so the model still sees plain background.
  - Test pages are NOT cropped: they are evaluated on full pages.

Usage:
  python scripts/make_crops.py --split manifests/split.csv --out manifests/crops.csv
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def seed_for(pair_id, seed):
    return int(hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--per-image", type=int, default=10)
    ap.add_argument("--min-std", type=float, default=0.04)
    ap.add_argument("--blank-frac", type=float, default=0.2)
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = [r for r in csv.DictReader(open(args.split)) if r["split"] in ("train", "val")]
    max_blank = int(round(args.per_image * args.blank_frac))
    rows, short = [], 0
    for k, r in enumerate(pairs, 1):
        with Image.open(r["target_path"]) as im:
            tgt = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        h, w = tgt.shape
        if h < args.size or w < args.size:
            continue
        rng = random.Random(seed_for(r["pair_id"], args.seed))
        got, blank = [], 0
        for _ in range(args.max_tries):
            if len(got) == args.per_image:
                break
            x, y = rng.randint(0, w - args.size), rng.randint(0, h - args.size)
            std = float(tgt[y:y + args.size, x:x + args.size].std())
            content = std >= args.min_std
            if not content:
                if blank >= max_blank:
                    continue
                blank += 1
            got.append((x, y, std, content))
        short += len(got) < args.per_image
        for i, (x, y, std, content) in enumerate(got):
            rows.append({"crop_id": f"{r['pair_id']}#{i}", "pair_id": r["pair_id"], "group": r["group"],
                         "split": r["split"], "input_path": r["input_path"], "target_path": r["target_path"],
                         "x": x, "y": y, "size": args.size, "target_std": round(std, 4), "has_content": int(content)})
        if k % 200 == 0:
            print(f"  {k}/{len(pairs)} pairs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {s: {"crops": n} for s, n in Counter(r["split"] for r in rows).items()}
    for s in summary:
        sel = [r for r in rows if r["split"] == s]
        summary[s]["pairs"] = len({r["pair_id"] for r in sel})
        summary[s]["page_groups"] = len({r["group"] for r in sel})
        summary[s]["content_fraction"] = round(sum(r["has_content"] for r in sel) / len(sel), 3)
    summary.update(size=args.size, per_image=args.per_image, min_std=args.min_std,
                   blank_frac=args.blank_frac, seed=args.seed, pairs_with_fewer_crops=short)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
