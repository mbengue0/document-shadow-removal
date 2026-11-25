"""Stage 3a: document-level train / validation / test split.

Why: the same printed page appears in many SD7K pairs (different shadows). If one page is in
both training and test, test scores are inflated. So we split by page group, not by image.

Rules (applied in this order):
  1. Only pairs with status "ok" in manifest.csv are used.
  2. Any page group that contains an official test pair goes to "test".
     Official training pairs in such a group are EXCLUDED (not moved), so the official test
     set stays unchanged and no page leaks. Their number is reported.
  3. Remaining groups (official training only) are shuffled with a fixed seed;
     about --val-frac of the groups become "val", the rest "train".

Group IDs come from build_manifest.py (candidate_group). Check them visually first.

Usage:
  python scripts/make_split.py --manifest manifests/manifest.csv --out manifests/split.csv
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def is_official_test(split_name):
    return "test" in split_name.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.manifest)) if r["status"] == "ok"]
    groups = defaultdict(list)
    for r in rows:
        groups[r["candidate_group"]].append(r)

    test_groups = {g for g, rs in groups.items() if any(is_official_test(r["split"]) for r in rs)}
    train_groups = sorted(set(groups) - test_groups)
    random.Random(args.seed).shuffle(train_groups)
    n_val = max(1, round(len(train_groups) * args.val_frac)) if train_groups else 0
    val_groups = set(train_groups[:n_val])

    out_rows, excluded = [], 0
    for g, rs in groups.items():
        for r in rs:
            if g in test_groups:
                if is_official_test(r["split"]):
                    new, reason = "test", "official test"
                else:
                    new, reason = "excluded", "official train pair sharing a page with test"
                    excluded += 1
            else:
                new = "val" if g in val_groups else "train"
                reason = "official train"
            out_rows.append({"pair_id": r["pair_id"], "group": g, "official_split": r["split"],
                             "split": new, "reason": reason, "input_path": r["input_path"],
                             "target_path": r["target_path"], "width": r["input_w"], "height": r["input_h"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: (r["split"], r["pair_id"])))

    summary = {}
    for name in ("train", "val", "test", "excluded"):
        sel = [r for r in out_rows if r["split"] == name]
        summary[name] = {"pairs": len(sel), "page_groups": len({r["group"] for r in sel})}
    summary.update(seed=args.seed, val_frac=args.val_frac, excluded_train_pairs=excluded)
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
