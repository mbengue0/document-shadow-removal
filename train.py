"""Train the U-Net on crops.csv.

Every experimental condition uses this same script; only --lambda-edge (and --seed) change.
The checkpoint is selected by validation L1 for every condition, so runs with different
loss weights are compared on the same criterion.

Examples:
  # pixel loss only
  python train.py --crops manifests/crops.csv --lambda-edge 0 --out runs/l1_seed0
  # pixel + edge loss
  python train.py --crops manifests/crops.csv --lambda-edge 0.5 --out runs/edge0.5_seed0
  # sanity check: can the model memorise one batch?
  python train.py --crops manifests/crops.csv --overfit-steps 300 --out runs/overfit
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import PairCrops, collate_crops
from src.losses import RestorationLoss
from src.model import UNet, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, amp_dtype):
    model.eval()
    sums = {"l1": 0.0, "edge": 0.0, "mse": 0.0, "input_l1": 0.0}
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = model(x)
        p = p.float()
        _, parts = loss_fn(p.clamp(0, 1), y)
        b = x.shape[0]
        sums["l1"] += parts["l1"].item() * b
        sums["edge"] += parts["edge"].item() * b
        sums["mse"] += ((p.clamp(0, 1) - y) ** 2).mean().item() * b
        sums["input_l1"] += (x - y).abs().mean().item() * b  # "do nothing" reference
        n += b
    model.train()
    out = {k: v / n for k, v in sums.items()}
    # PSNR of the pooled error over all validation crops. Averaging per-crop PSNR would be
    # dominated by shadow-free crops whose error is near zero (PSNR -> infinity).
    out["psnr"] = 10 * math.log10(1.0 / max(out.pop("mse"), 1e-10))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lambda-edge", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-images", type=int, default=2, help="image pairs per step (x crops per image)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="for pilot runs (0 = full epoch)")
    ap.add_argument("--overfit-steps", type=int, default=0, help="train on one fixed batch only")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if (args.amp == "none" or device.type != "cuda") else (
        torch.bfloat16 if args.amp == "bf16" else torch.float16)
    args.out.mkdir(parents=True, exist_ok=True)

    train_set, val_set = PairCrops(args.crops, "train"), PairCrops(args.crops, "val")
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_images, shuffle=True, generator=g,
                              num_workers=args.workers, collate_fn=collate_crops, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch_images, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_crops)

    model = UNet(base=args.base, depth=args.depth).to(device)
    loss_fn = RestorationLoss(args.lambda_edge)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    config = vars(args) | {"device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                           "parameters": count_parameters(model), "train_pairs": len(train_set),
                           "train_crops": train_set.n_crops, "val_pairs": len(val_set), "val_crops": val_set.n_crops,
                           "torch": torch.__version__}
    (args.out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(json.dumps(config, indent=2, default=str))

    def step(x, y):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = model(x)
        loss, parts = loss_fn(p.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return parts

    if args.overfit_steps:
        x, y = next(iter(train_loader))
        with open(args.out / "overfit_log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "l1", "edge", "total", "input_l1"])
            for s in range(1, args.overfit_steps + 1):
                parts = step(x, y)
                if s == 1 or s % 25 == 0:
                    row = [s] + [round(parts[k].item(), 5) for k in ("l1", "edge", "total")] + [round((x - y).abs().mean().item(), 5)]
                    w.writerow(row)
                    print("overfit", row, flush=True)
        return

    steps_per_epoch = len(train_loader) if not args.max_steps_per_epoch else min(len(train_loader), args.max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, args.epochs * steps_per_epoch) / (args.epochs * steps_per_epoch))))

    best, log_path = float("inf"), args.out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_l1", "train_edge", "train_total", "val_l1", "val_edge",
                                "val_psnr", "val_input_l1", "lr", "epoch_seconds", "peak_gpu_mb"])
    for epoch in range(1, args.epochs + 1):
        t0, sums, n = time.time(), {"l1": 0.0, "edge": 0.0, "total": 0.0}, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i, (x, y) in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            parts = step(x, y)
            sched.step()
            for k in sums:
                sums[k] += parts[k].item()
            n += 1
        val = evaluate(model, val_loader, loss_fn, device, amp_dtype)
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else 0
        row = [epoch] + [round(sums[k] / n, 5) for k in ("l1", "edge", "total")] + [
            round(val["l1"], 5), round(val["edge"], 5), round(val["psnr"], 3), round(val["input_l1"], 5),
            f"{sched.get_last_lr()[0]:.2e}", round(secs, 1), round(peak)]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print("epoch", row, flush=True)
        state = {"model": model.state_dict(), "epoch": epoch, "val": val, "config": config}
        torch.save(state, args.out / "last.pt")
        if val["l1"] < best:
            best = val["l1"]
            torch.save(state, args.out / "best.pt")


if __name__ == "__main__":
    main()
