import argparse, os, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

#Train a small CNN to classify math vs text patches using context crops from full page images.

# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 2),
        )
    def forward(self, x): return self.net(x)


# ---------- dataset ----------
PATCH_RE = re.compile(r"^(?P<page>.+)__(?P<x>\d+)_(?P<y>\d+)_(?P<grid>\d+)__(?P<label>math|text)\.png$", re.IGNORECASE)

def index_pages(pages_dir: Path) -> Dict[str, Path]:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
    mp = {}
    for p in pages_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mp[p.stem] = p
    return mp

def crop_context(page_img: Image.Image, x: int, y: int, grid: int, context: int) -> Image.Image:
    W, H = page_img.size
    cx = x + grid / 2.0
    cy = y + grid / 2.0
    half = context / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + context
    y2 = y1 + context

    # pad if out of bounds
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    if pad_left or pad_top or pad_right or pad_bottom:
        # make padded image
        newW = W + pad_left + pad_right
        newH = H + pad_top + pad_bottom
        canvas = Image.new("RGB", (newW, newH), (255, 255, 255))
        canvas.paste(page_img, (pad_left, pad_top))
        x1 += pad_left
        y1 += pad_top
        page_img = canvas

    return page_img.crop((x1, y1, x1 + context, y1 + context))

class ContextPatchDataset(Dataset):
    def __init__(self, pages_dir: str, patches_root: str, context: int, input_size: int):
        self.pages_dir = Path(pages_dir)
        self.patches_root = Path(patches_root)
        self.context = int(context)
        self.input_size = int(input_size)

        self.page_index = index_pages(self.pages_dir)
        if not self.page_index:
            raise RuntimeError(f"No page images found in {self.pages_dir}")

        self.items: List[Tuple[str, int, int, int, int]] = []  # (page_stem, x,y,grid,label_idx)
        # label_idx: 0=math, 1=text (fixed order)
        for label_name, label_idx in [("math", 0), ("text", 1)]:
            folder = self.patches_root / label_name
            if not folder.exists():
                raise RuntimeError(f"Missing folder: {folder}")
            for fp in folder.glob("*.png"):
                m = PATCH_RE.match(fp.name)
                if not m:
                    continue
                page = m.group("page")
                x = int(m.group("x")); y = int(m.group("y")); grid = int(m.group("grid"))
                if page not in self.page_index:
                    # skip if page image not found
                    continue
                self.items.append((page, x, y, grid, label_idx))

        if not self.items:
            raise RuntimeError("No usable patch items found. Check filenames + pages_dir matching stems.")

        self.tfm = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # small page cache (avoids re-opening file constantly)
        self._cache_page: str = ""
        self._cache_img: Image.Image = None

    def __len__(self): return len(self.items)

    def _load_page(self, stem: str) -> Image.Image:
        if stem == self._cache_page and self._cache_img is not None:
            return self._cache_img
        img = Image.open(self.page_index[stem]).convert("RGB")
        self._cache_page = stem
        self._cache_img = img
        return img

    def __getitem__(self, idx):
        stem, x, y, grid, label = self.items[idx]
        page = self._load_page(stem)
        crop = crop_context(page, x, y, grid, self.context)
        x_tensor = self.tfm(crop)
        return x_tensor, label


def page_level_split(items: List[Tuple[str,int,int,int,int]], val_frac: float, seed: int):
    # split by page stem
    rng = np.random.default_rng(seed)
    pages = sorted({it[0] for it in items})
    rng.shuffle(pages)
    n_val = max(1, int(len(pages) * val_frac))
    val_pages = set(pages[:n_val])
    tr_idx, va_idx = [], []
    for i, it in enumerate(items):
        (va_idx if it[0] in val_pages else tr_idx).append(i)
    return tr_idx, va_idx, len(val_pages), len(pages) - len(val_pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_dir", required=True)
    ap.add_argument("--patches_dir", required=True, help=".../dataset (contains math/ text/)")
    ap.add_argument("--out", default="run_ctx")
    ap.add_argument("--grid", type=int, default=48, help="label grid size encoded in filename (48)")
    ap.add_argument("--context", type=int, default=96, help="context crop size from full page")
    ap.add_argument("--input_size", type=int, default=96, help="CNN input resize size")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_pages", type=float, default=0.2, help="fraction of pages for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    ds = ContextPatchDataset(args.pages_dir, args.patches_dir, context=args.context, input_size=args.input_size)
    # filter to only those matching grid.
    ds.items = [it for it in ds.items if it[3] == args.grid]
    if not ds.items:
        raise RuntimeError(f"No items with grid={args.grid}. Check your filenames.")
    print("items:", len(ds.items), "context:", args.context, "grid:", args.grid)

    tr_idx, va_idx, n_val_pages, n_tr_pages = page_level_split(ds.items, args.val_pages, args.seed)
    print("page split -> train pages:", n_tr_pages, "val pages:", n_val_pages)
    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    va_ds = torch.utils.data.Subset(ds, va_idx)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # weighted loss to boost recall
    # label 0=math, 1=text
    math_count = sum(1 for (_,_,_,_,lab) in ds.items if lab == 0)
    text_count = sum(1 for (_,_,_,_,lab) in ds.items if lab == 1)
    total = math_count + text_count
    w_math = total / max(1, math_count)
    w_text = total / max(1, text_count)
    class_weights = torch.tensor([w_math, w_text], dtype=torch.float32).to(device)
    print("counts math/text:", math_count, text_count, "loss weights:", class_weights.tolist())

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "cnn_math_text.pt"
    best_va = 0.0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_correct = tr_total = 0
        for x, y in tqdm(tr_loader, desc=f"ep {ep} train"):
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += x.size(0)

        model.eval()
        va_correct = va_total = 0
        with torch.no_grad():
            for x, y in tqdm(va_loader, desc=f"ep {ep} val"):
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)
                logits = model(x)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += x.size(0)

        tr_acc = tr_correct / max(1, tr_total)
        va_acc = va_correct / max(1, va_total)
        print(f"ep {ep}: train acc {tr_acc:.4f} | val acc {va_acc:.4f}")

        if va_acc > best_va:
            best_va = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": ["math", "text"],
                "grid": args.grid,
                "context": args.context,
                "input_size": args.input_size,
            }, best_path)
            print("  saved:", best_path)

    print("best val acc:", best_va, "model:", best_path)

if __name__ == "__main__":
    main()
