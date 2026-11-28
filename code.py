"""
Document Shadow Removal with Deep Learning and Text Preservation
CS-4212 Deep Learning and Neural Networks - Mouhamadou Mbengue

Research question
    Does adding an edge-preservation term to a pixel reconstruction loss improve document
    shadow removal while preserving text details?

What this single file does (run:  python code.py)
    1. Finds the SD7K dataset (downloads and unzips it if it is missing).
    2. Groups photos by printed page, splits by page (no page in two sets) and selects the
       256 x 256 training/validation crops. The result is written to dataset.csv.
       If dataset.csv already exists next to this file, it is reused as is.
    3. Checks every image pair (decoding, size, alignment) and copies the crops into a fast
       on-disk cache.
    4. Trains the same U-Net under three loss settings: L1 only (lambda = 0) and
       L1 + lambda * edge loss for two values of lambda. Training resumes automatically
       if it was interrupted.
    5. Tunes a classical (non-learning) shadow correction on validation photos.
    6. Restores every full-resolution test photo (overlapping tiles), computes PSNR, SSIM,
       L1 and a text-edge error, compares methods per page with bootstrap confidence
       intervals, and draws the figures.
    7. Optional: evaluates phone photos placed in <data>/phone (see PHONE PHOTOS below).

Requirements: Python 3.10+, torch, numpy, pillow, opencv-python-headless, matplotlib.
Missing packages are installed automatically. A CUDA GPU is strongly recommended for
training (about 1 hour per condition on a Colab T4); evaluation also runs on CPU, slowly.

Useful options
    python code.py --quick            small end-to-end run (a few minutes) to check that
                                      everything works; its numbers are NOT results
    python code.py --data PATH        folder that contains SD7K/train and SD7K/test
    python code.py --out PATH         where checkpoints, results and figures are written

PHONE PHOTOS (optional): put paired photos in <data>/phone/input and <data>/phone/target
(same file names) and unpaired photos in <data>/phone/unpaired. They are used for testing
only, never for training.

AI assistance disclosure
    In accordance with the course AI usage policy, an AI assistant (Claude, by Anthropic)
    was used to help write and debug this code, check the dataset, and structure the
    experiment. All design decisions, the review of the code, and the interpretation of the
    results are the author's responsibility.

References
    Li, Z., Chen, X., Pun, C.-M., and Cun, X. (2023). High-Resolution Document Shadow Removal
    via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net. ICCV.
    Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional Networks for
    Biomedical Image Segmentation. MICCAI.
"""

# ----------------------------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------------------------
import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path


def _ensure(pkg, pip_name=None):
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pip_name or pkg} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or pkg])


for _p, _pip in [("numpy", None), ("PIL", "pillow"), ("cv2", "opencv-python-headless"),
                 ("matplotlib", None), ("torch", None)]:
    _ensure(_p, _pip)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent

# Shared Google Drive copy of the SD7K zip (the official OneDrive link is also listed in
# the SD7K repository: https://github.com/CXH-Research/DocShadow-SD7K).
DATA_ZIP_URL = ""  # filled in with the shared Drive link before submission

CONFIG = {
    "seed": 0,
    "crop_size": 256,
    "crops_per_image": 10,
    "min_content_std": 0.04,   # a crop "has text" if the grey-level std of its clean target >= this
    "max_blank_frac": 0.2,     # at most 20 % of an image's crops may be near-blank
    "val_frac": 0.10,          # share of training PAGES used for validation
    "group_threshold": 0.95,   # thumbnail correlation above which two clean photos are the same page
    "max_shift_px": 2,         # alignment tolerance, in pixels of a 512-pixel-wide copy
    "base_channels": 32,
    "depth": 4,
    "epochs": 6,
    "batch_size": 16,
    "lr": 2e-4,
    # Edge weights, fixed BEFORE training: on 200 random training crops the shadowed input
    # differs from its target by L1 = 0.152 and edge error = 0.0052 (ratio about 29).
    # lambda = 7 makes the edge term about a quarter of the pixel term, lambda = 30 about equal.
    "lambdas": [0.0, 7.0, 30.0],
    "tile": 256,
    "tile_overlap": 64,
    "text_threshold": 0.5,     # clean-target luminance below this = text/graphics pixel
    "bootstrap": 2000,
}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


# ----------------------------------------------------------------------------------------
# 1. Data location
# ----------------------------------------------------------------------------------------
def find_sd7k(data_dir):
    """Folder that contains train/input and test/input, searched up to three levels down."""
    if not data_dir.exists():
        return None
    for cand in [data_dir, *sorted(p for p in data_dir.glob("*/") if p.is_dir()),
                 *sorted(p for p in data_dir.glob("*/*/") if p.is_dir()),
                 *sorted(p for p in data_dir.glob("*/*/*/") if p.is_dir())]:
        if (cand / "train" / "input").is_dir() and (cand / "test" / "input").is_dir():
            return cand
    return None


def ensure_data(data_dir):
    root = find_sd7k(data_dir)
    if root:
        return root
    if not DATA_ZIP_URL:
        sys.exit(f"SD7K not found under {data_dir}. Download it from "
                 "https://github.com/CXH-Research/DocShadow-SD7K and unzip it there "
                 "(expected folders: SD7K/train/input, SD7K/train/target, SD7K/test/...).")
    _ensure("gdown")
    import gdown
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "SD7K.zip"
    if not zpath.exists():
        log("Downloading SD7K (about 9 GB) ...")
        gdown.download(DATA_ZIP_URL, str(zpath), quiet=False, fuzzy=True)
    log("Unzipping ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_dir)
    root = find_sd7k(data_dir)
    if not root:
        sys.exit("Unzipped, but could not find SD7K/train and SD7K/test folders.")
    return root


def list_pairs(root):
    """All (split, name) with both an input and a target file. Unmatched files are reported."""
    pairs, unmatched = [], []
    for split in ("train", "test"):
        inp = {p.name for p in (root / split / "input").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        tgt = {p.name for p in (root / split / "target").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        unmatched += [f"{split}/{'input' if n in inp else 'target'}/{n}" for n in sorted(inp ^ tgt)]
        pairs += [(split, n) for n in sorted(inp & tgt)]
    return pairs, unmatched


# ----------------------------------------------------------------------------------------
# 2. Page groups, split and crop selection  ->  dataset.csv
# ----------------------------------------------------------------------------------------
def _target_pass(job):
    """Read one clean target: small thumbnail (for page grouping) and crop positions."""
    root, split, name, size, k, min_std, max_blank, seed = job
    tgt = np.asarray(Image.open(root / split / "target" / name).convert("L"), dtype=np.float32) / 255.0
    h, w = tgt.shape
    thumb = np.asarray(Image.fromarray((tgt * 255).astype(np.uint8)).resize((48, 64), Image.BILINEAR), np.float32)
    rng = random.Random(int(hashlib.sha256(f"{seed}:{split}/{name}".encode()).hexdigest()[:12], 16))
    crops, blank = [], 0
    for _ in range(300):
        if len(crops) == k or w < size or h < size:
            break
        x, y = rng.randint(0, w - size), rng.randint(0, h - size)
        std = float(tgt[y:y + size, x:x + size].std())
        if std < min_std:
            if blank >= max_blank:
                continue
            blank += 1
        crops.append((x, y, round(std, 4)))
    return split, name, w, h, thumb, crops


def page_groups(thumbs, threshold):
    """Union-find over photos whose clean-target thumbnails correlate above threshold."""
    t = np.stack(thumbs).reshape(len(thumbs), -1)
    t = t - t.mean(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-8
    parent = list(range(len(t)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, len(t), 1024):
        sim = t[s:s + 1024] @ t.T
        for a, b in zip(*np.nonzero(sim >= threshold)):
            if s + a < b:
                parent[find(s + a)] = find(b)
    roots = [find(i) for i in range(len(t))]
    order = {r: f"g{k:04d}" for k, r in enumerate(sorted(set(roots), key=roots.index))}
    return [order[r] for r in roots]


def build_dataset_csv(root, pairs, path, cfg, workers):
    log(f"Pass 1/2: reading {len(pairs)} clean targets (page groups and crop positions) ...")
    jobs = [(root, s, n, cfg["crop_size"], cfg["crops_per_image"], cfg["min_content_std"],
             int(round(cfg["crops_per_image"] * cfg["max_blank_frac"])), cfg["seed"]) for s, n in pairs]
    with Pool(workers) as pool:
        res = list(pool.imap(_target_pass, jobs, chunksize=8))
    groups = page_groups([r[4] for r in res], cfg["group_threshold"])

    # Split by page: a page containing any official test photo is a test page; official
    # training photos of test pages are excluded so that no test page is ever seen in
    # training. About val_frac of the remaining (training-only) pages become validation.
    test_pages = {g for g, r in zip(groups, res) if r[0] == "test"}
    train_pages = sorted({g for g in groups} - test_pages)
    random.Random(cfg["seed"]).shuffle(train_pages)
    val_pages = set(train_pages[:max(1, round(len(train_pages) * cfg["val_frac"]))])

    rows = []
    for g, (split, name, w, h, _, crops) in zip(groups, res):
        rel_in, rel_tg = f"{split}/input/{name}", f"{split}/target/{name}"
        if g in test_pages:
            new = "test" if split == "test" else "excluded"
            rows.append(dict(split=new, page=g, unit="page", input=rel_in, target=rel_tg,
                             x="", y="", size="", width=w, height=h, target_std="", official_split=split))
            continue
        new = "val" if g in val_pages else "train"
        for x, y, std in crops:
            rows.append(dict(split=new, page=g, unit="crop", input=rel_in, target=rel_tg, x=x, y=y,
                             size=cfg["crop_size"], width=w, height=h, target_std=std, official_split=split))
    for i, r in enumerate(rows):
        r["row_id"] = i
    write_csv(path, rows, ["row_id", "split", "unit", "page", "input", "target", "x", "y", "size",
                           "width", "height", "target_std", "official_split"])
    log(f"Wrote {path} ({len(rows)} rows)")
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------------------
# 3. Pair checks and crop cache
# ----------------------------------------------------------------------------------------
def _edge_small(a):
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    s = 512.0 / max(g.shape)
    g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def estimate_shift(a, b):
    """Translation between two images (pixels of a 512-wide copy), by phase correlation of edges."""
    ea, eb = _edge_small(a), _edge_small(b)
    win = cv2.createHanningWindow(ea.shape[::-1], cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ea, eb, win)
    return dx, dy


def _pair_pass(job):
    root, inp_rel, tgt_rel, crops, size = job
    try:
        a, b = load_rgb(root / inp_rel), load_rgb(root / tgt_rel)
    except Exception as e:  # unreadable file
        return inp_rel, "decode_error:" + repr(e)[:80], None, None, 0.0, 0.0
    if a.shape != b.shape:
        return inp_rel, "size_mismatch", None, None, 0.0, 0.0
    dx, dy = estimate_shift(a, b)
    xa = np.stack([a[y:y + size, x:x + size] for x, y in crops]) if crops else None
    xb = np.stack([b[y:y + size, x:x + size] for x, y in crops]) if crops else None
    return inp_rel, "ok", xa, xb, dx, dy


def build_cache(root, rows, cache_dir, out_dir, cfg, workers):
    """Check every pair; copy the crops of good pairs into memory-mapped arrays."""
    done = cache_dir / "cache_index.json"
    if done.exists():
        return json.loads(done.read_text())
    cache_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["crop_size"]
    by_pair = defaultdict(list)
    for r in rows:
        if r["split"] in ("train", "val", "test"):
            by_pair[(r["input"], r["target"])].append(r)
    n_crops = sum(1 for r in rows if r["unit"] == "crop")
    X = np.lib.format.open_memmap(cache_dir / "inputs.npy", "w+", np.uint8, (n_crops, size, size, 3))
    Y = np.lib.format.open_memmap(cache_dir / "targets.npy", "w+", np.uint8, (n_crops, size, size, 3))
    jobs = [(root, i, t, [(int(r["x"]), int(r["y"])) for r in rs if r["unit"] == "crop"], size)
            for (i, t), rs in by_pair.items()]
    log(f"Pass 2/2: checking {len(jobs)} pairs and caching {n_crops} crops ...")
    checks, index, k = [], {"train": [], "val": []}, 0
    crop_rows = {r["input"]: [c for c in by_pair[(r["input"], r["target"])] if c["unit"] == "crop"] for r in rows}
    with Pool(workers) as pool:
        for j, (inp_rel, status, xa, xb, dx, dy) in enumerate(pool.imap(_pair_pass, jobs, chunksize=4)):
            if status == "ok" and max(abs(dx), abs(dy)) > cfg["max_shift_px"]:
                status = "misaligned"
            checks.append(dict(input=inp_rel, status=status, shift_x=round(dx, 2), shift_y=round(dy, 2)))
            if status == "ok" and xa is not None:
                n = len(xa)
                X[k:k + n], Y[k:k + n] = xa, xb
                for c in crop_rows[inp_rel]:
                    index[c["split"]].append(k)
                    k += 1
            if (j + 1) % 500 == 0:
                log(f"  {j + 1}/{len(jobs)} pairs")
    X.flush()
    Y.flush()
    write_csv(out_dir / "pair_checks.csv", checks, ["input", "status", "shift_x", "shift_y"])
    bad = {c["input"] for c in checks if c["status"] != "ok"}
    index["bad_pairs"] = sorted(bad)
    index["n_filled"] = k
    done.write_text(json.dumps(index))
    log(f"Pair checks: {len(checks) - len(bad)} ok, {len(bad)} rejected (see pair_checks.csv)")
    return index


class CachedCrops(Dataset):
    def __init__(self, cache_dir, idx):
        self.cache_dir, self.idx = cache_dir, np.asarray(idx)
        self.X = self.Y = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.X is None:  # open lazily in each worker
            self.X = np.load(self.cache_dir / "inputs.npy", mmap_mode="r")
            self.Y = np.load(self.cache_dir / "targets.npy", mmap_mode="r")
        j = self.idx[i]
        x = torch.from_numpy(np.array(self.X[j])).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy(np.array(self.Y[j])).permute(2, 0, 1).float().div_(255)
        return x, y


# ----------------------------------------------------------------------------------------
# 4. Model: compact U-Net for RGB restoration
# ----------------------------------------------------------------------------------------
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) adapted to restoration: RGB in, RGB out, same size.
    GroupNorm (independent of batch size), and a residual output: the network predicts a
    correction added to the input. The last layer starts at zero, so the untrained network
    returns the input unchanged. Identical for all experimental conditions."""

    def __init__(self, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.depth = depth
        self.down = nn.ModuleList([conv_block(3, ch[0])] + [conv_block(ch[i - 1], ch[i]) for i in range(1, depth + 1)])
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.up = nn.ModuleList([conv_block(2 * ch[i - 1], ch[i - 1]) for i in range(depth, 0, -1)])
        self.head = nn.Conv2d(ch[0], 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips, h = [], x
        for i, block in enumerate(self.down):
            h = block(h)
            if i < self.depth:
                skips.append(h)
                h = F.max_pool2d(h, 2)
        for upc, block, s in zip(self.upconv, self.up, reversed(skips)):
            h = block(torch.cat([upc(h), s], 1))
        return x + self.head(h)


# ----------------------------------------------------------------------------------------
# 5. Losses and metrics
# ----------------------------------------------------------------------------------------
SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
LUMA = torch.tensor([0.299, 0.587, 0.114])


def luminance(x):
    return (x * LUMA.to(x).view(1, 3, 1, 1)).sum(1, keepdim=True)


def sobel(y):
    """Sobel gradients (kernels / 8, replicate padding) of a 1-channel image batch."""
    k = SOBEL.to(y).view(1, 1, 3, 3)
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(y, k), F.conv2d(y, k.transpose(2, 3))


def edge_map_error(pred, target):
    """Per-pixel |grad_x difference| + |grad_y difference|, halved (luminance)."""
    gxp, gyp = sobel(luminance(pred))
    gxt, gyt = sobel(luminance(target))
    return 0.5 * ((gxp - gxt).abs() + (gyp - gyt).abs())


def loss_fn(pred, target, lam):
    l1 = (pred - target).abs().mean()
    edge = edge_map_error(pred, target).mean()
    return l1 + lam * edge, l1.detach(), edge.detach()


def ssim_luma(a, b):
    """SSIM on luminance, 11x11 Gaussian window (sigma 1.5), data range 1."""
    x, y = luminance(a), luminance(b)
    g = torch.exp(-((torch.arange(11, dtype=x.dtype, device=x.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum()).view(1, 1, 1, 11)

    def blur(z):
        return F.conv2d(F.conv2d(z, g), g.transpose(2, 3))

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return s.mean().item()


def page_metrics(pred, target, text_thr):
    """pred, target: 1x3xHxW tensors in [0, 1]."""
    pred = pred.clamp(0, 1)
    mse = ((pred - target) ** 2).mean().item()
    text = luminance(target) < text_thr
    text = F.max_pool2d(text.float(), 5, stride=1, padding=2) > 0  # include stroke borders
    em = edge_map_error(pred, target)
    return {"psnr": 10 * math.log10(1 / max(mse, 1e-10)), "ssim": ssim_luma(pred, target),
            "l1": (pred - target).abs().mean().item(),
            "text_edge_err": (em[text].mean().item() if text.any() else float("nan")),
            "text_fraction": text.float().mean().item()}


# ----------------------------------------------------------------------------------------
# 6. Training
# ----------------------------------------------------------------------------------------
def use_amp(device):
    """Mixed precision only on GPUs with fast float16 (compute capability >= 7.0, e.g. T4).
    Older cards (e.g. Quadro P1000) are much slower in float16, so they use float32."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7


def run_name(lam):
    return "unet_l1" if lam == 0 else f"unet_l1_edge{lam:g}"


@torch.no_grad()
def validate(model, loader, device, amp):
    model.eval()
    s = {"l1": 0.0, "edge": 0.0, "mse": 0.0, "n": 0}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            p = model(x)
        p = p.float().clamp(0, 1)
        b = len(x)
        s["l1"] += (p - y).abs().mean().item() * b
        s["edge"] += edge_map_error(p, y).mean().item() * b
        s["mse"] += ((p - y) ** 2).mean().item() * b
        s["n"] += b
    model.train()
    n = s.pop("n")
    out = {k: v / n for k, v in s.items()}
    out["psnr"] = 10 * math.log10(1 / max(out.pop("mse"), 1e-10))  # PSNR of the pooled error
    return out


def train_condition(lam, cache_dir, index, out_dir, cfg, device, workers):
    rdir = out_dir / "runs" / run_name(lam)
    rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "done.json").exists():
        log(f"{run_name(lam)}: already trained, skipping")
        return
    set_seed(cfg["seed"])  # same initial weights and batch order for every condition
    model = UNet(cfg["base_channels"], cfg["depth"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    amp = use_amp(device)
    scaler = torch.amp.GradScaler(enabled=amp)
    g = torch.Generator().manual_seed(cfg["seed"])
    tl = DataLoader(CachedCrops(cache_dir, index["train"]), batch_size=cfg["batch_size"], shuffle=True,
                    generator=g, num_workers=workers, pin_memory=device.type == "cuda", drop_last=True,
                    persistent_workers=workers > 0)
    vl = DataLoader(CachedCrops(cache_dir, index["val"]), batch_size=64, num_workers=workers)
    total = cfg["epochs"] * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    start, best, history = 1, float("inf"), []
    if (rdir / "last.pt").exists():  # resume after an interruption
        st = torch.load(rdir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best, history = st["epoch"] + 1, st["best"], st["history"]
        torch.set_rng_state(st["rng"])
        g.set_state(st["loader_rng"])
        log(f"{run_name(lam)}: resuming at epoch {start}")
    log(f"{run_name(lam)}: lambda={lam}, {len(tl.dataset)} train crops, {len(vl.dataset)} val crops, "
        f"{sum(p.numel() for p in model.parameters()):,} parameters")
    for epoch in range(start, cfg["epochs"] + 1):
        t0, sums, n = time.time(), [0.0, 0.0, 0.0], 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                p = model(x)
            loss, l1, edge = loss_fn(p.float(), y, lam)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums[0] += loss.item()
            sums[1] += l1.item()
            sums[2] += edge.item()
            n += 1
        val = validate(model, vl, device, amp)
        rec = {"epoch": epoch, "train_total": sums[0] / n, "train_l1": sums[1] / n, "train_edge": sums[2] / n,
               "val_l1": val["l1"], "val_edge": val["edge"], "val_psnr": val["psnr"],
               "seconds": round(time.time() - t0, 1),
               "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else 0}
        history.append(rec)
        log(f"{run_name(lam)} epoch {epoch}/{cfg['epochs']}: " + ", ".join(
            f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items() if k != "epoch"))
        if val["l1"] < best:  # same selection rule (validation L1) for every condition
            best = val["l1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "lambda": lam}, rdir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history,
                    "rng": torch.get_rng_state(), "loader_rng": g.get_state()}, rdir / "last.pt")
        write_csv(rdir / "history.csv", history, list(rec))
    (rdir / "done.json").write_text(json.dumps({"best_val_l1": best, "epochs": cfg["epochs"]}))
    (rdir / "last.pt").unlink(missing_ok=True)  # keep only best.pt to save space


# ----------------------------------------------------------------------------------------
# 7. Classical baseline: background estimation and division
# ----------------------------------------------------------------------------------------
def classical_correct(img, k, paper):
    """Estimate the page background per channel on a 1/4-size copy with a morphological
    closing (removes dark text) and a median filter, then divide the image by it.
    img: HxWx3 uint8. k: kernel size at 1/4 scale. paper: RGB level of clean paper."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    bg = cv2.medianBlur(bg, 21)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) + 1.0
    out = img.astype(np.float32) / bg * np.asarray(paper, np.float32).reshape(1, 1, 3) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def tune_classical(root, rows, out_dir, n_pages=40, seed=0):
    path = out_dir / "classical_params.json"
    if path.exists():
        return json.loads(path.read_text())
    val_pairs = sorted({(r["input"], r["target"]) for r in rows if r["split"] == "val"})
    random.Random(seed).shuffle(val_pairs)
    val_pairs = val_pairs[:n_pages]
    imgs = [(load_rgb(root / a), load_rgb(root / b)) for a, b in val_pairs]
    # paper level: median colour of the brighter half of clean validation pixels
    tg = np.concatenate([b.reshape(-1, 3)[::50] for _, b in imgs]).astype(np.float32) / 255
    lum = tg @ np.array([0.299, 0.587, 0.114], np.float32)
    paper = np.median(tg[lum >= np.median(lum)], axis=0).tolist()
    scores = {}
    for k in (11, 21, 41, 81):
        scores[k] = float(np.mean([np.abs(classical_correct(a, k, paper).astype(np.float32) - b).mean() / 255
                                   for a, b in imgs]))
        log(f"classical baseline: kernel {k} -> validation L1 {scores[k]:.4f}")
    best = min(scores, key=scores.get)
    params = {"kernel": best, "paper": paper, "val_l1": scores, "n_val_photos": len(imgs)}
    path.write_text(json.dumps(params, indent=2))
    return params


# ----------------------------------------------------------------------------------------
# 8. Full-page inference and evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def restore_page(model, img, device, tile, overlap, batch=32):
    """Restore a full photo with overlapping tiles blended by a smooth window."""
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)
    _, _, h, w = x.shape
    step = tile - overlap
    ph, pw = max(tile, math.ceil((h - tile) / step) * step + tile), max(tile, math.ceil((w - tile) / step) * step + tile)
    xp = F.pad(x, (0, pw - w, 0, ph - h), mode="reflect" if (pw - w < w and ph - h < h) else "replicate")
    win1 = torch.hann_window(tile, periodic=False).clamp_min(1e-3)
    win = (win1[:, None] * win1[None, :]).view(1, 1, tile, tile)
    out, wsum = torch.zeros(1, 3, ph, pw), torch.zeros(1, 1, ph, pw)
    coords = [(i, j) for i in range(0, ph - tile + 1, step) for j in range(0, pw - tile + 1, step)]
    amp = use_amp(device)
    for s in range(0, len(coords), batch):
        cs = coords[s:s + batch]
        tiles = torch.cat([xp[:, :, i:i + tile, j:j + tile] for i, j in cs]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pr = model(tiles)
        pr = pr.float().cpu()
        for (i, j), t in zip(cs, pr):
            out[:, :, i:i + tile, j:j + tile] += t * win
            wsum[:, :, i:i + tile, j:j + tile] += win
    return (out / wsum)[:, :, :h, :w].clamp(0, 1)


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0)


def to_u8(t):
    return (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def evaluate(root, pages, models, classical, out_dir, cfg, device, tag="sd7k", save_examples=None):
    """pages: list of (page_group, input_rel, target_rel). Writes per-photo metrics."""
    path = out_dir / f"results_{tag}.csv"
    done = {}
    if path.exists():
        for r in read_csv(path):
            done[(r["input"], r["method"])] = r
    fields = ["page", "input", "method", "psnr", "ssim", "l1", "text_edge_err", "text_fraction"]
    rows = list(done.values())
    mdev = {k: m.to(device).eval() for k, m in models.items()}
    t0 = time.time()
    for n, (page, a_rel, b_rel) in enumerate(pages):
        if all((a_rel, m) in done for m in ["input", "classical", *models]):
            continue
        a, b = load_rgb(root / a_rel), load_rgb(root / b_rel)
        ta, tb = to_t(a), to_t(b)
        tb_d = tb.to(device)
        preds = {"input": ta, "classical": to_t(classical_correct(a, classical["kernel"], classical["paper"]))}
        for k, m in mdev.items():
            preds[k] = restore_page(m, a, device, cfg["tile"], cfg["tile_overlap"])
        for k, p in preds.items():
            m = page_metrics(p.to(device), tb_d, cfg["text_threshold"])
            rows.append({"page": page, "input": a_rel, "method": k, **{q: round(v, 6) for q, v in m.items()}})
        if save_examples is not None and a_rel in save_examples:
            np.savez_compressed(out_dir / "examples" / (Path(a_rel).stem + f"_{tag}.npz"), target=b,
                                **{k: to_u8(p) for k, p in preds.items()})
        if (n + 1) % 20 == 0 or n + 1 == len(pages):
            write_csv(path, rows, fields)
            log(f"evaluated {n + 1}/{len(pages)} photos ({(time.time() - t0) / 60:.1f} min)")
    write_csv(path, rows, fields)
    return rows


def summarize(rows, methods, out_dir, cfg, tag="sd7k", reference="unet_l1"):
    """Average per page (photos of one page are not independent), then over pages.
    95 % bootstrap intervals over pages; paired differences against the reference."""
    metrics = ["psnr", "ssim", "l1", "text_edge_err"]
    per_page = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for q in metrics:
            v = float(r[q])
            if not math.isnan(v):
                per_page[(r["page"], r["method"])][q].append(v)
    pages = sorted({r["page"] for r in rows})
    rng = np.random.default_rng(cfg["seed"])
    boot = rng.integers(0, len(pages), size=(cfg["bootstrap"], len(pages)))

    def page_vec(m, q):
        return np.array([np.mean(per_page[(p, m)][q]) for p in pages])

    summary = {"n_pages": len(pages), "n_photos": len({r["input"] for r in rows}), "methods": {}, "vs_" + reference: {}}
    for m in methods:
        summary["methods"][m] = {}
        for q in metrics:
            v = page_vec(m, q)
            bs = v[boot].mean(1)
            summary["methods"][m][q] = {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    if reference in methods:
        for m in methods:
            if m == reference:
                continue
            summary["vs_" + reference][m] = {}
            for q in metrics:
                d = page_vec(m, q) - page_vec(reference, q)
                bs = d[boot].mean(1)
                summary["vs_" + reference][m][q] = {
                    "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "pages_better": int((d < 0).sum() if q in ("l1", "text_edge_err") else (d > 0).sum())}
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    lines = [f"Results on {summary['n_photos']} photos of {summary['n_pages']} pages ({tag}); "
             "mean over pages [95% bootstrap CI]", "",
             "| Method | PSNR (dB) up | SSIM up | L1 down | Text edge error down |", "|---|---|---|---|---|"]
    for m in methods:
        s = summary["methods"][m]
        lines.append(f"| {m} | " + " | ".join(
            f"{s[q]['mean']:.{2 if q == 'psnr' else 4}f} [{s[q]['ci95'][0]:.{2 if q == 'psnr' else 4}f}, "
            f"{s[q]['ci95'][1]:.{2 if q == 'psnr' else 4}f}]" for q in metrics) + " |")
    (out_dir / f"summary_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return summary


# ----------------------------------------------------------------------------------------
# 9. Figures
# ----------------------------------------------------------------------------------------
LABELS = {"input": "Shadowed input", "classical": "Classical", "unet_l1": "U-Net L1"}


def label(m):
    return LABELS.get(m, m.replace("unet_l1_edge", "U-Net L1+edge λ="))


def fig_training(out_dir, lambdas):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for lam in lambdas:
        p = out_dir / "runs" / run_name(lam) / "history.csv"
        if not p.exists():
            continue
        h = read_csv(p)
        e = [int(r["epoch"]) for r in h]
        ax[0].plot(e, [float(r["val_l1"]) for r in h], marker="o", label=label(run_name(lam)))
        ax[1].plot(e, [float(r["val_edge"]) for r in h], marker="o", label=label(run_name(lam)))
    ax[0].set(title="Validation L1 (lower is better)", xlabel="epoch")
    ax[1].set(title="Validation edge error (lower is better)", xlabel="epoch")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig_training_curves.png", dpi=150)
    plt.close(fig)


def fig_metrics(summary, methods, out_dir, tag):
    qs = [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("text_edge_err", "Text edge error ↓")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, (q, t) in zip(ax, qs):
        means = [summary["methods"][m][q]["mean"] for m in methods]
        lo = [means[i] - summary["methods"][m][q]["ci95"][0] for i, m in enumerate(methods)]
        hi = [summary["methods"][m][q]["ci95"][1] - means[i] for i, m in enumerate(methods)]
        a.bar(range(len(methods)), means, yerr=[lo, hi], capsize=4, color="#2F5496")
        a.set_xticks(range(len(methods)))
        a.set_xticklabels([label(m) for m in methods], rotation=25, ha="right", fontsize=8)
        a.set_title(t)
        a.grid(axis="y", alpha=0.3)
        if q != "text_edge_err":
            a.set_ylim(min(means) * 0.95 if q == "psnr" else max(0, min(means) - 0.05), None)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_metrics_{tag}.png", dpi=150)
    plt.close(fig)


def fig_examples(out_dir, methods, tag, crop=600):
    files = sorted((out_dir / "examples").glob(f"*_{tag}.npz"))
    if not files:
        return
    cols = methods + ["target"]
    fig, ax = plt.subplots(len(files), len(cols), figsize=(2.3 * len(cols), 2.5 * len(files)), squeeze=False)
    for r, f in enumerate(files):
        d = np.load(f)
        # Region chosen from input and target only (largest shadow change times text
        # density), never from the methods' outputs, so it is not picked by results.
        a, b = d["input"].astype(np.float32), d["target"].astype(np.float32)
        score = cv2.blur(np.abs(a - b).mean(2) * (b.mean(2) < 128), (crop, crop))
        h, w = score.shape
        y, x = np.unravel_index(np.argmax(score[crop // 2:h - crop // 2, crop // 2:w - crop // 2]),
                                (h - crop, w - crop)) if h > crop and w > crop else (0, 0)
        for c, k in enumerate(cols):
            ax[r, c].imshow(d[k][y:y + crop, x:x + crop])
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title("Clean target" if k == "target" else label(k), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_examples_{tag}.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# 10. Phone photos (test only)
# ----------------------------------------------------------------------------------------
def phone_eval(data_root, models, classical, out_dir, cfg, device, methods):
    phone = data_root.parent / "phone" if not (data_root / "phone").exists() else data_root / "phone"
    if not phone.exists():
        log("No phone photos found (optional), skipping.")
        return
    (out_dir / "phone").mkdir(exist_ok=True)
    if (phone / "input").is_dir() and (phone / "target").is_dir():
        names = sorted({p.name for p in (phone / "input").iterdir()} & {p.name for p in (phone / "target").iterdir()})
        pages = [(f"phone_{n}", f"input/{n}", f"target/{n}") for n in names]
        (out_dir / "examples").mkdir(exist_ok=True)
        rows = evaluate(phone, pages, models, classical, out_dir, cfg, device, tag="phone",
                        save_examples={p[1] for p in pages[:4]})
        s = summarize(rows, methods, out_dir, cfg, tag="phone")
        fig_metrics(s, methods, out_dir, "phone")
        fig_examples(out_dir, methods, "phone")
    if (phone / "unpaired").is_dir():
        for p in sorted((phone / "unpaired").iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            a = load_rgb(p)
            outs = [a, classical_correct(a, classical["kernel"], classical["paper"])]
            outs += [to_u8(restore_page(m.to(device).eval(), a, device, cfg["tile"], cfg["tile_overlap"]))
                     for m in models.values()]
            s = 1000 / max(a.shape[:2])
            row = np.concatenate([cv2.resize(o, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for o in outs], 1)
            Image.fromarray(row).save(out_dir / "phone" / f"{p.stem}_comparison.jpg", quality=90)
        log("Unpaired phone photos: comparisons saved in outputs/phone "
            "(order: input, classical, " + ", ".join(models) + ")")


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--cache", type=Path, default=None, help="crop cache folder (fast local disk)")
    ap.add_argument("--dataset-csv", type=Path, default=HERE / "dataset.csv")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny end-to-end check; numbers are not results")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.epochs:
        cfg["epochs"] = args.epochs
    out = args.out / ("quick" if args.quick else "")
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or out / "cache"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    log(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}; output: {out}")

    root = ensure_data(args.data)
    pairs, unmatched = list_pairs(root)
    log(f"SD7K at {root}: {len(pairs)} complete pairs; unmatched files: {unmatched or 'none'}")

    # dataset.csv: reuse the shipped one, otherwise build it (and keep a copy in outputs)
    ds_path = args.dataset_csv
    if args.quick:
        rng = random.Random(0)
        pairs = rng.sample([p for p in pairs if p[0] == "train"], 60) + rng.sample([p for p in pairs if p[0] == "test"], 6)
        cfg.update(epochs=1, base_channels=16, crops_per_image=4, val_frac=0.3, bootstrap=200)
        ds_path = out / "dataset_quick.csv"
        rows = read_csv(ds_path) if ds_path.exists() else build_dataset_csv(root, pairs, ds_path, cfg, args.workers)
    elif ds_path.exists() or (out / "dataset.csv").exists():
        ds_path = ds_path if ds_path.exists() else out / "dataset.csv"
        rows = read_csv(ds_path)
        log(f"Using existing {ds_path} ({len(rows)} rows)")
    else:
        rows = build_dataset_csv(root, pairs, out / "dataset.csv", cfg, args.workers)
    counts = defaultdict(lambda: [0, set(), set()])
    for r in rows:
        c = counts[r["split"]]
        c[0] += 1
        c[1].add(r["page"])
        c[2].add(r["input"])
    stats = {s: {"rows": c[0], "pages": len(c[1]), "photos": len(c[2])} for s, c in counts.items()}
    log("Dataset: " + json.dumps(stats))
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    index = build_cache(root, rows, cache, out, cfg, args.workers)
    bad = set(index["bad_pairs"])

    models = {}
    for lam in cfg["lambdas"]:
        train_condition(lam, cache, index, out, cfg, device, min(args.workers, 4))
        m = UNet(cfg["base_channels"], cfg["depth"])
        m.load_state_dict(torch.load(out / "runs" / run_name(lam) / "best.pt", map_location="cpu",
                                     weights_only=False)["model"])
        models[run_name(lam)] = m
    fig_training(out, cfg["lambdas"])

    classical = tune_classical(root, rows, out)
    log(f"Classical baseline: kernel {classical['kernel']}, paper level {np.round(classical['paper'], 3).tolist()}")

    test = sorted({(r["page"], r["input"], r["target"]) for r in rows if r["split"] == "test" and r["input"] not in bad})
    first_photo = {}
    for p, a, _ in test:
        first_photo.setdefault(p, a)
    examples = set(sorted(first_photo.values())[:4])  # first photo of the first 4 test pages
    (out / "examples").mkdir(exist_ok=True)
    methods = ["input", "classical", *models]
    res = evaluate(root, test, models, classical, out, cfg, device, save_examples=examples)
    summary = summarize(res, methods, out, cfg)
    fig_metrics(summary, methods, out, "sd7k")
    fig_examples(out, methods, "sd7k")
    phone_eval(root, models, classical, out, cfg, device, methods)
    log(f"Done. Results, figures and checkpoints are in {out}")


if __name__ == "__main__":
    main()
