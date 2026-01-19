#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional
import re
import numpy as np
import cv2
from PIL import Image, ImageOps

import torch
from transformers import AutoProcessor

# Qwen2.5-VL class name can vary slightly depending on your transformers version.
# Try import; fall back with a helpful error if missing.
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception as e:
    Qwen2_5_VLForConditionalGeneration = None
    _QWEN_IMPORT_ERR = e


Box = Tuple[int, int, int, int]  # (x, y, w, h)


def read_boxes_csv(csv_path: Path) -> List[Box]:
    boxes: List[Box] = []
    if not csv_path.exists():
        return boxes
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            w = int(float(row["w"]))
            h = int(float(row["h"]))
            if w > 1 and h > 1:
                boxes.append((x, y, w, h))
    return boxes


def autocrop_to_ink(pil_img: Image.Image, pad: int = 6, min_ink_frac: float = 0.002) -> Image.Image:
    """
    Crops tightly around dark ink pixels. Reused from your script to keep behavior consistent.
    """
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = gray < 200

    H, W = ink.shape
    ink_count = int(ink.sum())
    if ink_count < int(min_ink_frac * H * W):
        return pil_img

    ys, xs = np.where(ink)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)

    return pil_img.crop((x0, y0, x1, y1))


def clamp_box(b: Box, W: int, H: int, pad_x: int, pad_y: int) -> Optional[Tuple[int, int, int, int]]:
    x, y, w, h = b
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(W, x + w + pad_x)
    y1 = min(H, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def load_page_image(page_name: str, img_root: Path, img_exts: List[str]) -> Optional[Path]:
    for ext in img_exts:
        p = img_root / f"{page_name}{ext}"
        if p.exists():
            return p
    for ext in img_exts:
        cand = list(img_root.glob(f"{page_name}{ext}"))
        if cand:
            return cand[0]
    return None


def resize_if_small(img: Image.Image, min_h: int = 96, max_h: int = 768) -> Image.Image:
    w, h = img.size
    if h < min_h:
        scale = float(min_h) / max(1, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.BICUBIC)
    if h > max_h:
        scale = float(max_h) / h
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.BICUBIC)
    return img


def invert_if_dark(img_rgb: Image.Image, thresh: float = 90.0) -> Image.Image:
    gray = ImageOps.grayscale(img_rgb)
    mean = float(np.array(gray).mean())
    if mean < thresh:
        return ImageOps.invert(img_rgb)
    return img_rgb


def sanitize_plain_math(s: str) -> str:
    """
    Clean up model output for Markdown-friendly plain text.
    We keep it conservative: never delete content aggressively.
    """
    s = (s or "").strip()
    if not s:
        return s

    # Remove common wrapper chatter if it appears
    # (Some VLMs echo system instructions or add labels.)
    s = re.sub(r"^(transcription|result|ocr)\s*:\s*", "", s, flags=re.I).strip()

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # Hard cap super-long junk; keep the beginning
    if len(s) > 500:
        s = s[:500].rstrip() + " …"
    return s


DEFAULT_PROMPT = (
    "You are an OCR engine for handwritten math.\n"
    "Transcribe ONLY the math expression in the image.\n"
    "Output Markdown-friendly plain text math (ASCII). Do NOT output LaTeX.\n"
    "Use simple operators: + - * / ^ _ = ( ) [ ] { } , .\n"
    "If uncertain, output your best guess rather than empty.\n"
    "Return only the transcription, no extra words."
    "Once you identify the math expression, wrap it with proper $ delimiters.\n"
)


class QwenMathOCR:
    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True):
        if Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError(
                f"Failed to import Qwen2_5_VLForConditionalGeneration from transformers: {_QWEN_IMPORT_ERR}\n"
                "Fix: upgrade transformers (and accelerate), or install from source."
            )
        self.device = device
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)

        dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda" if device.startswith("cuda") else None,
        )
        self.model.eval()

    @torch.no_grad()
    def infer_one(self, img: Image.Image, prompt: str, max_new_tokens: int = 128) -> str:
        # Qwen2.5-VL expects chat format with an explicit image placeholder.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Important: apply_chat_template injects the image tokens.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )

        if self.device.startswith("cuda"):
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

        # Decode only the generated continuation (avoid echo)
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        text_out = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        text_out = sanitize_plain_math(text_out)
        return text_out if text_out else "[unreadable-math]"

        def infer_wrap_math_only(self, text: str, max_new_tokens: int = 256) -> str:
            system = (
                "You are a deterministic text tagger. "
                "You must not perform OCR, reasoning, translation, or formatting. "
                "You must not output LaTeX commands such as \\\\text, \\\\frac, \\\\sum."
            )

            prompt = f"""Task: Insert math delimiters ONLY.

        Rules (strict):
        - You may add ONLY the character '$'.
        - You may not add any other characters.
        - You may not remove or modify any characters.
        - You may not rewrite or explain anything.
        - Do not use LaTeX commands.
        - Do not add $$ unless the entire line is mathematical.
        - Do not nest or duplicate delimiters.
        - If a math expression is already wrapped, leave it unchanged.

        Examples:

        Input:
        Verifier proves that f(x) = 0 for all x ∈ X.
        Output:
        Verifier proves that $f(x) = 0$ for all $x ∈ X$.

        Input:
        f(x) = x^2 + 1
        Output:
        $$
        f(x) = x^2 + 1
        $$

        Input:
        We assume $x ≥ 0$.
        Output:
        We assume $x ≥ 0$.

        Now process this text and return ONLY the processed text:
        {text}
        """

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            # call Qwen with text-only messages
            # return decoded_text



def main():
    # Keep arguments compatible with your current run_math_ocr_from_boxes.py :contentReference[oaicite:1]{index=1}
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks_dir", required=True, help="blocks/ directory containing per-page folders")
    ap.add_argument("--img_root", required=True,
                    help="Root folder containing original page images named <page_name>.<ext>")
    ap.add_argument("--img_exts", default=".png,.jpg,.jpeg",
                    help="Comma-separated extensions to try, e.g. .png,.jpg")
    ap.add_argument("--out_root", required=True, help="Where to write math crops + csv results")

    ap.add_argument("--pad_x", type=int, default=6, help="Padding around math crop in pixels (x)")
    ap.add_argument("--pad_y", type=int, default=4, help="Padding around math crop in pixels (y)")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing <page>_math_ocr.csv")
    ap.add_argument("--skip_existing_crops", action="store_true",
                    help="If crop file exists, reuse it instead of re-cropping (useful when iterating OCR)")

    # New optional args (safe defaults; won’t break existing calls)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct",
                    help="HF model name for Qwen2.5-VL. Default fits 12GB GPUs well.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for OCR.")
    ap.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens generated per crop.")
    ap.add_argument("--min_h", type=int, default=96, help="Upscale crops smaller than this height.")
    ap.add_argument("--max_h", type=int, default=768, help="Downscale crops taller than this height.")
    ap.add_argument("--invert_if_dark", action="store_true", help="Invert crop if background is dark-ish.")
    args = ap.parse_args()

    blocks_dir = Path(args.blocks_dir)
    img_root = Path(args.img_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img_exts = [e.strip() for e in args.img_exts.split(",") if e.strip()]
    if not img_exts:
        raise SystemExit("No --img_exts provided")

    page_dirs = sorted([p for p in blocks_dir.iterdir() if p.is_dir()])
    if not page_dirs:
        raise SystemExit(f"No page folders found in {blocks_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] CUDA not available; CPU inference will be slow.")

    print(f"[INFO] Loading Qwen2.5-VL model: {args.model} on {device}")
    ocr = QwenMathOCR(args.model, device=device, fp16=(device == "cuda"))

    for page_dir in page_dirs:
        page_name = page_dir.name

        img_path = load_page_image(page_name, img_root, img_exts)
        if img_path is None:
            print(f"[SKIP] {page_name}: cannot find image under {img_root} with exts {img_exts}")
            continue

        inline_csv = page_dir / f"{page_name}_math_boxes_inline.csv"
        disp_csv = page_dir / f"{page_name}_math_boxes_display.csv"
        inline_boxes = read_boxes_csv(inline_csv)
        disp_boxes = read_boxes_csv(disp_csv)

        if not inline_boxes and not disp_boxes:
            print(f"[OK] {page_name}: no math boxes")
            continue

        out_page = out_root / page_name
        crops_dir = out_page / "math"
        crops_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_page / f"{page_name}_math_ocr.csv"
        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] {page_name}: {out_csv} exists")
            continue

        page_img = Image.open(img_path).convert("RGB")
        W, H = page_img.size

        tasks: List[Tuple[int, str, Box, Path]] = []
        mid = 0
        for kind, boxes in [("inline", inline_boxes), ("display", disp_boxes)]:
            for b in boxes:
                crop_path = crops_dir / f"math_{mid:04d}_{kind}.png"
                tasks.append((mid, kind, b, crop_path))
                mid += 1

        rows = []
        for (mid_, kind_, b_, crop_path) in tasks:
            # crop (or reuse existing crop file)
            if (not args.skip_existing_crops) or (not crop_path.exists()):
                clamped = clamp_box(b_, W, H, pad_x=args.pad_x, pad_y=args.pad_y)
                if clamped is None:
                    continue
                x0, y0, x1, y1 = clamped
                crop = page_img.crop((x0, y0, x1, y1))
                crop = autocrop_to_ink(crop, pad=6)
                crop = resize_if_small(crop, min_h=args.min_h, max_h=args.max_h)
                crop = crop.convert("RGB")
                if args.invert_if_dark:
                    crop = invert_if_dark(crop)
                crop.save(crop_path)
            else:
                crop = Image.open(crop_path).convert("RGB")
                crop = resize_if_small(crop, min_h=args.min_h, max_h=args.max_h)
                if args.invert_if_dark:
                    crop = invert_if_dark(crop)

            text = ocr.infer_one(crop, args.prompt, max_new_tokens=args.max_new_tokens)
            # text = ocr.infer_wrap_math_only(text, max_new_tokens=args.max_new_tokens)

            x, y, w, h = b_
            # Keep the column name "latex" to stay compatible with your downstream code
            rows.append([mid_, kind_, x, y, w, h, str(crop_path), text])

        out_page.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["math_id", "kind", "x", "y", "w", "h", "crop_path", "latex"])
            for r in rows:
                w.writerow(r)

        print(f"[OK] {page_name}: math={len(rows)} -> {out_csv}")

        if device == "cuda":
            torch.cuda.empty_cache()

    print("done.")


if __name__ == "__main__":
    main()
