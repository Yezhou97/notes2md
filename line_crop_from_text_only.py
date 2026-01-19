#!/usr/bin/env python3
"""
line_crop_from_text_only.py

Input layout:
  blocks/
    page_name/
      page_name_text_only.png

Output (per page_name/):
  lines/
    page_name_line_000.png
    page_name_line_001.png
    ...
  page_name_text_lines.csv
  page_name_text_lines_overlay.png
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ----------------------------
# Image preprocessing helpers
# ----------------------------

def binarize_ink(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )


def remove_ruled_lines(ink: np.ndarray, strength: int = 25) -> np.ndarray:
    if strength <= 0:
        return ink
    k = max(5, int(strength))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    horiz = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.subtract(ink, horiz)


# ----------------------------
# Line detection
# ----------------------------

def find_text_line_bands(
    ink: np.ndarray,
    row_frac_thr: float = 0.010,
    min_band_h: int = 10,
    merge_gap: int = 6,
) -> List[Tuple[int, int]]:
    H, W = ink.shape
    proj = (ink > 0).sum(axis=1)

    row_thr = max(8, int(row_frac_thr * W))
    active = proj > row_thr

    bands = []
    y = 0
    while y < H:
        if not active[y]:
            y += 1
            continue
        y0 = y
        while y < H and active[y]:
            y += 1
        y1 = y
        if (y1 - y0) >= min_band_h:
            bands.append((y0, y1))

    merged = []
    for (a0, a1) in bands:
        if not merged:
            merged.append([a0, a1])
        else:
            b0, b1 = merged[-1]
            if a0 - b1 <= merge_gap:
                merged[-1][1] = a1
            else:
                merged.append([a0, a1])

    return [(int(a0), int(a1)) for a0, a1 in merged]


def crop_lines(
    img: np.ndarray,
    bands: List[Tuple[int, int]],
    pad_top: int,
    pad_bot: int,
) -> List[np.ndarray]:
    H, W = img.shape[:2]
    out = []
    for y0, y1 in bands:
        yy0 = max(0, y0 - pad_top)
        yy1 = min(H, y1 + pad_bot)
        out.append(img[yy0:yy1, 0:W])
    return out


# ----------------------------
# Per-image processing
# ----------------------------

def process_text_only_image(
    img_path: Path,
    args,
):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[FAIL] cannot read {img_path}")
        return

    # page_name from blocks/page_name/page_name_text_only.png
    page_name = img_path.parent.name
    base = page_name  # canonical base name

    # output location
    Path(args.blocks_dir).mkdir(exist_ok=True)
    page_dir = Path(args.blocks_dir) / page_name
    page_dir.mkdir(parents=True, exist_ok=True)

    lines_dir = Path(args.lines_dir) / page_name
    lines_dir.mkdir(parents=True, exist_ok=True)

    manifest = page_dir / f"{base}_text_lines.csv"
    if manifest.exists() and not args.overwrite:
        print(f"[SKIP] {manifest}")
        return

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ink = binarize_ink(gray)

    if args.remove_rules:
        ink = remove_ruled_lines(ink, args.rule_strength)

    bands = find_text_line_bands(
        ink,
        row_frac_thr=args.row_frac_thr,
        min_band_h=args.min_band_h,
        merge_gap=args.merge_gap,
    )

    crops = crop_lines(
        bgr,
        bands,
        pad_top=args.pad_top,
        pad_bot=args.pad_bot,
    )

    # Write crops + CSV
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "y0", "y1", "path"])
        for i, ((y0, y1), crop) in enumerate(zip(bands, crops)):
            out_path = lines_dir / f"{base}_line_{i:03d}_text.png"
            cv2.imwrite(str(out_path), crop)
            w.writerow([i, y0, y1, str(out_path)])

    # Debug overlay
    dbg = bgr.copy()
    for y0, y1 in bands:
        cv2.rectangle(dbg, (0, y0), (dbg.shape[1] - 1, y1), (0, 0, 255), 1)
    cv2.imwrite(str(page_dir / f"{base}_text_lines_overlay.png"), dbg)

    print(f"[OK] {base}: {len(bands)} lines")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks_dir", required=True,
                    help="Path to blocks/ directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--lines_dir", required=True)

    # Line detection
    ap.add_argument("--row_frac_thr", type=float, default=0.010)
    ap.add_argument("--min_band_h", type=int, default=10)
    ap.add_argument("--merge_gap", type=int, default=6)
    ap.add_argument("--pad_top", type=int, default=4)
    ap.add_argument("--pad_bot", type=int, default=4)

    # Ruled lines
    ap.add_argument("--remove_rules", action="store_true")
    ap.add_argument("--rule_strength", type=int, default=25)

    args = ap.parse_args()

    blocks_dir = Path(args.blocks_dir)
    imgs = sorted(blocks_dir.glob("*/*_text_only.png"))

    if not imgs:
        raise SystemExit("No *_text_only.png found under blocks/")

    for img in imgs:
        process_text_only_image(img, args)

    print("done.")


if __name__ == "__main__":
    main()
