"""Dataset that serves the crops listed in crops.csv.

Each item is ONE image pair with all of its listed crops, so a 2462 x 3699 photo is decoded
once per epoch instead of once per crop. The collate function flattens them into one batch:
batch_images pairs x crops_per_image crops = crops per optimisation step.

Input and target are cut at exactly the same (x, y); no geometric augmentation is applied
in the first version, so the pair cannot become misaligned.
"""
import csv
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def to_tensor(a):
    return torch.from_numpy(a).permute(2, 0, 1).float().div_(255.0)


class PairCrops(Dataset):
    def __init__(self, crops_csv, split):
        by_pair = defaultdict(list)
        for r in csv.DictReader(open(crops_csv)):
            if r["split"] == split:
                by_pair[r["pair_id"]].append(r)
        if not by_pair:
            raise ValueError(f"No crops for split '{split}' in {crops_csv}")
        self.pairs = sorted(by_pair.items())
        self.n_crops = sum(len(v) for _, v in self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        _, rows = self.pairs[i]
        inp, tgt = load_rgb(rows[0]["input_path"]), load_rgb(rows[0]["target_path"])
        if inp.shape != tgt.shape:
            raise ValueError(f"Size mismatch in pair {rows[0]['pair_id']}")
        xs, ys = [], []
        for r in rows:
            x, y, s = int(r["x"]), int(r["y"]), int(r["size"])
            xs.append(to_tensor(inp[y:y + s, x:x + s]))
            ys.append(to_tensor(tgt[y:y + s, x:x + s]))
        return torch.stack(xs), torch.stack(ys)


def collate_crops(items):
    return torch.cat([a for a, _ in items]), torch.cat([b for _, b in items])
