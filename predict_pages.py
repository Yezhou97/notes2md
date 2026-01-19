import argparse, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm


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

def list_images(path: Path) -> List[Path]:
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}
    if path.is_file():
        return [path] if path.suffix.lower() in exts else []
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts])

def iter_grid(W: int, H: int, grid: int):
    for y in range(0, H, grid):
        for x in range(0, W, grid):
            w = min(grid, W - x)
            h = min(grid, H - y)
            yield x, y, w, h

def crop_context(page: Image.Image, x: int, y: int, grid: int, context: int) -> Image.Image:
    W, H = page.size
    cx = x + grid / 2.0
    cy = y + grid / 2.0
    half = context / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + context
    y2 = y1 + context

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    if pad_left or pad_top or pad_right or pad_bottom:
        canvas = Image.new("RGB", (W + pad_left + pad_right, H + pad_top + pad_bottom), (255,255,255))
        canvas.paste(page, (pad_left, pad_top))
        x1 += pad_left
        y1 += pad_top
        page = canvas

    return page.crop((x1, y1, x1 + context, y1 + context))

def add_red_tint(rgb: np.ndarray, x: int, y: int, w: int, h: int, strength: int = 90):
    rgb[y:y+h, x:x+w, 0] = np.clip(rgb[y:y+h, x:x+w, 0].astype(np.int16) + strength, 0, 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="one image file OR folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=48, help="grid size on page")
    ap.add_argument("--context", type=int, default=96, help="context crop size")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--overlay_strength", type=int, default=90)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device)
    input_size = int(ckpt.get("input_size", 96))
    model = SmallCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5]),
    ])

    in_path = Path(args.input)
    imgs = list_images(in_path)
    if not imgs:
        raise SystemExit(f"No images found: {in_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(imgs, desc="predict"):
        page = Image.open(img_path).convert("RGB")
        W, H = page.size
        overlay = np.array(page, dtype=np.uint8).copy()
        mask = np.zeros((H, W), dtype=np.uint8)

        rows = []
        batch_t = []
        batch_meta: List[Tuple[int,int,int,int]] = []

        def flush():
            nonlocal batch_t, batch_meta, rows, mask, overlay
            if not batch_t:
                return
            X = torch.cat(batch_t, dim=0).to(device, non_blocking=True)
            with torch.no_grad():
                p = torch.softmax(model(X), dim=1)[:, 0].detach().cpu().numpy()  # class0=math
            for (x,y,w,h), pm in zip(batch_meta, p):
                pred = float(pm) >= args.thr
                if pred:
                    mask[y:y+h, x:x+w] = 255
                    add_red_tint(overlay, x, y, w, h, args.overlay_strength)
                rows.append([str(img_path), x, y, w, h, args.grid, args.context, int(pred), float(pm)])
            batch_t = []
            batch_meta = []

        for (x, y, w, h) in iter_grid(W, H, args.grid):
            crop = crop_context(page, x, y, args.grid, args.context)
            xt = tfm(crop).unsqueeze(0)
            batch_t.append(xt)
            batch_meta.append((x,y,w,h))
            if len(batch_t) >= args.batch:
                flush()
        flush()

        base = img_path.stem
        cv2.imwrite(str(out_dir / f"{base}_pred_math_mask.png"), mask)
        cv2.imwrite(str(out_dir / f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        with open(out_dir / f"{base}_pred_patches.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["img_path","x","y","w","h","grid","context","pred_math","p_math"])
            w.writerows(rows)

if __name__ == "__main__":
    main()
