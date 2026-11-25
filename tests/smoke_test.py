"""End-to-end code check on SYNTHETIC images (not project data, never reported as results).

Creates fake document pages with a dark triangular "shadow", then runs:
  build_manifest -> make_split -> make_crops -> unit checks -> overfit check -> 2 short epochs.
It only shows that the code runs and can learn; it says nothing about real performance.

Usage: python tests/smoke_test.py   (CPU is enough, about a few minutes)
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.losses import RestorationLoss, edge_loss  # noqa: E402
from src.model import UNet, count_parameters  # noqa: E402


def fake_page(seed, w=320, h=416):
    rng = np.random.default_rng(seed)
    im = Image.new("L", (w, h), 235)
    d = ImageDraw.Draw(im)
    for y in range(20, h - 20, 14):
        x = 16
        while x < w - 40:
            ww = int(rng.integers(6, 28))
            d.rectangle([x, y, x + ww, y + 6], fill=30)
            x += ww + 6
    return np.asarray(im, dtype=np.float32)


def add_shadow(a, k):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx / w + yy / h) > (0.9 + 0.2 * k)
    return np.where(mask, a * 0.5, a)


def save(a, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGB").save(path)


def run(*cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], check=True, cwd=ROOT)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    data, man = tmp / "data", tmp / "manifests"
    i = 0
    for page in range(12):  # 12 fake pages, 3 shadows each
        split = "test" if page >= 10 else "train"
        clean = fake_page(page)
        for k in range(3):
            save(add_shadow(clean, k), data / split / "input" / f"{i:04d}.png")
            save(clean, data / split / "target" / f"{i:04d}.png")
            i += 1

    run("scripts/build_manifest.py", "--root", data, "--out", man)
    run("scripts/make_split.py", "--manifest", man / "manifest.csv", "--out", man / "split.csv", "--val-frac", "0.2")
    run("scripts/make_crops.py", "--split", man / "split.csv", "--out", man / "crops.csv", "--size", "64", "--per-image", "4")

    # leakage check: no page group in more than one split
    split_of = {}
    for r in csv.DictReader(open(man / "split.csv")):
        if r["split"] != "excluded":
            assert split_of.setdefault(r["group"], r["split"]) == r["split"], "page group in two splits"
    print("leakage check passed:", len(split_of), "page groups")

    # model and loss checks
    m = UNet(base=16, depth=3)
    x = torch.rand(2, 3, 64, 64)
    assert m(x).shape == x.shape
    print("U-Net (base 16, depth 3) parameters:", count_parameters(m))
    print("U-Net (base 32, depth 4) parameters:", count_parameters(UNet()))
    assert edge_loss(x, x).item() == 0.0
    loss, parts = RestorationLoss(0.5)(m(x), torch.rand(2, 3, 64, 64))
    loss.backward()
    assert all(p.grad is not None for p in m.parameters()), "missing gradients"
    print("forward/backward check passed", {k: round(v.item(), 4) for k, v in parts.items()})

    run("train.py", "--crops", man / "crops.csv", "--out", tmp / "runs/overfit", "--base", "16", "--depth", "3",
        "--workers", "0", "--overfit-steps", "300", "--lambda-edge", "0.5", "--lr", "1e-3")
    run("train.py", "--crops", man / "crops.csv", "--out", tmp / "runs/short", "--base", "16", "--depth", "3",
        "--workers", "0", "--epochs", "2", "--lambda-edge", "0.5")
    print("\nAll checks ran. Outputs in", tmp)


if __name__ == "__main__":
    main()
