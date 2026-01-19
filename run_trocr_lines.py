#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import re

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def parse_line_id(p: Path) -> int:
    s = p.stem  # e.g. "PCP (1)_line_000_text"
    m = re.search(r"(?:^|[_-])line[_-]?(\d+)", s)
    if m:
        return int(m.group(1))
    # last resort: last number in stem
    nums = re.findall(r"(\d+)", s)
    return int(nums[-1]) if nums else -1

def natural_sort_key(p: Path):
    # sort ..._line_001.png numerically
    s = p.stem
    # expect suffix _line_XXX
    try:
        idx = parse_line_id(p)
    except Exception:
        idx = 10**9
    return (p.parent.name, idx, p.name)


@torch.inference_mode()
def trocr_batch(
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    images: List[Image.Image],
    device: torch.device,
    max_new_tokens: int,
    num_beams: int,
) -> List[str]:
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,

        no_repeat_ngram_size=3,     # prevents repeating 3-grams
        repetition_penalty=1.15,    # discourages repeats
        length_penalty=0.9,
    )
    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # light cleanup
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines_root", required=True, help="Root dir, e.g. lines/")
    ap.add_argument("--model", default="microsoft/trocr-large-handwritten",
                    help="HF model id, e.g. microsoft/trocr-large-handwritten")
    ap.add_argument("--glob", default="*_line_*.png", help="Pattern inside each page folder")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=2)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing *_trocr.csv")
    args = ap.parse_args()

    lines_root = Path(args.lines_root)
    if not lines_root.exists():
        raise SystemExit(f"lines_root not found: {lines_root}")

    # choose device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] loading model={args.model} on device={device}")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Find page folders under lines_root
    page_dirs = sorted([p for p in lines_root.iterdir() if p.is_dir()])
    if not page_dirs:
        raise SystemExit(f"No page folders under {lines_root}")

    for page_dir in page_dirs:
        imgs = sorted(page_dir.glob(args.glob), key=natural_sort_key)
        if not imgs:
            continue

        page_name = page_dir.name
        out_csv = page_dir / f"{page_name}_trocr.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] {out_csv} exists")
            continue

        print(f"[INFO] {page_name}: {len(imgs)} lines")

        rows: List[Tuple[str, int, str]] = []
        batch_paths = []
        batch_imgs = []

        for p in imgs:
            line_counter = 0
            batch_paths.append(p)
            batch_imgs.append(Image.open(p).convert("RGB"))

            if len(batch_imgs) >= args.batch_size:
                texts = trocr_batch(processor, model, batch_imgs, device,
                                   max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)

                for pp, txt in zip(batch_paths, texts):
                    rows.append((str(pp), line_counter, txt))
                    line_counter += 1

                batch_paths, batch_imgs = [], []

        # leftover
        if batch_imgs:
            texts = trocr_batch(processor, model, batch_imgs, device,max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            for pp, txt in zip(batch_paths, texts):
                try:
                    # line_id = int(pp.stem.split("_line_")[-1])
                    line_id = parse_line_id(pp)
                except Exception:
                    line_id = -1
                rows.append((str(pp), line_id, txt))

        # write CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "line_id", "text"])
            for r in rows:
                w.writerow(list(r))

        print(f"[OK] wrote {out_csv}")

    print("done.")


if __name__ == "__main__":
    main()
