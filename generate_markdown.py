#!/usr/bin/env python3
"""
generate_markdown_from_text_math.py

Merges:
- TrOCR line text (from lines/<page>/<page>_trocr.csv)
- line bands (from lines/<page>/<page>_text_lines.csv)
- pix2tex math OCR (from math_ocr/<page>/<page>_math_ocr.csv)
into a per-page Markdown file.

Folder assumptions (customizable via args):
  lines_root/<page_name>/
    <page_name>_text_lines.csv        (line_id,y0,y1,path)
    <page_name>_trocr.csv             (image_path,line_id,text)
  math_root/<page_name>/
    <page_name>_math_ocr.csv          (math_id,kind,x,y,w,h,crop_path,latex)
  img_root/
    <page_name>.png|.jpg|.jpeg        (optional but recommended for width W)

Output:
  out_root/<page_name>.md

Notes:
- Inline math insertion is approximate (by x-position mapped to character index).
- Display math becomes its own $$ ... $$ block.
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image


def load_csv_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def try_load_page_width(page_name: str, img_root: Path, exts=(".png", ".jpg", ".jpeg")) -> Optional[int]:
    for ext in exts:
        p = img_root / f"{page_name}{ext}"
        if p.exists():
            with Image.open(p) as im:
                return int(im.size[0])
    return None


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def insert_inline_segments(text: str, segments: List[Tuple[int, str]], page_w: Optional[int]) -> str:
    """
    segments: list of (x, latex_inline) sorted by x
    Insert latex into text approximately by x-position -> char index.
    If page_w is None or text is empty, append at end.
    """
    text = normalize_text(text)

    if not segments:
        return text

    if not text or not page_w or page_w <= 0:
        # fallback: append in order
        parts = [text] if text else []
        for _, seg in segments:
            if seg:
                parts.append(seg)
        return normalize_text(" ".join(parts))

    chars = list(text)
    L = len(chars)

    # build insert operations (index, token)
    ops: List[Tuple[int, str]] = []
    for x, seg in segments:
        if not seg:
            continue
        frac = max(0.0, min(1.0, float(x) / float(page_w)))
        idx = int(round(frac * L))
        idx = max(0, min(L, idx))
        ops.append((idx, seg))

    # stable insert from left to right with offset
    ops.sort(key=lambda t: t[0])
    out = []
    last = 0
    offset = 0
    for idx, seg in ops:
        idx2 = idx  # idx on original
        out.append("".join(chars[last:idx2]))
        # spacing heuristics
        if out and out[-1] and not out[-1].endswith(" "):
            out.append(" ")
        out.append(seg)
        out.append(" ")
        last = idx2
        offset += 1
    out.append("".join(chars[last:]))

    merged = "".join(out)
    # fix spaces around punctuation
    merged = re.sub(r"\s+([,.;:!?])", r"\1", merged)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged


def find_line_for_math(cy: float, lines: List[dict]) -> Optional[int]:
    """
    lines: list of {line_id, y0, y1, ...}
    Return line_id whose [y0,y1] contains cy.
    """
    for ln in lines:
        y0 = float(ln["y0"]); y1 = float(ln["y1"])
        if y0 <= cy <= y1:
            return int(ln["line_id"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks_root", required=True, help="Root with per-page folders")
    ap.add_argument("--lines_root", required=True, help="Root with per-page line outputs")
    ap.add_argument("--math_root", required=True, help="Root with per-page math_ocr outputs")
    ap.add_argument("--out_root", required=True, help="Where to write <page>.md")
    ap.add_argument("--img_root", default="", help="Optional: original page images for width-based insertion")
    ap.add_argument("--img_exts", default=".png,.jpg,.jpeg", help="Extensions to try for page images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    blocks_root = Path(args.blocks_root)
    lines_root = Path(args.lines_root)
    math_root = Path(args.math_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    img_root = Path(args.img_root) if args.img_root else None
    img_exts = tuple(e.strip() for e in args.img_exts.split(",") if e.strip())

    page_dirs = sorted([p for p in lines_root.iterdir() if p.is_dir()])
    if not page_dirs:
        raise SystemExit(f"No page folders under {lines_root}")

    for page_dir in page_dirs:
        page_name = page_dir.name
        out_md = out_root / f"{page_name}.md"
        if out_md.exists() and not args.overwrite:
            print(f"[SKIP] {out_md} exists")
            continue

        # Inputs
        blocks_page = Path(args.blocks_root) / page_name
        lines_csv = blocks_page / f"{page_name}_text_lines.csv"
        trocr_csv = page_dir / f"{page_name}_trocr.csv"
        math_csv = math_root / page_name / f"{page_name}_math_ocr.csv"

        lines_rows = load_csv_dicts(lines_csv)
        trocr_rows = load_csv_dicts(trocr_csv)
        math_rows = load_csv_dicts(math_csv)
        
        if not lines_rows or not trocr_rows:
            print(f"[SKIP] {page_name}: missing lines/trocr csv")
            continue

        # page width (optional)
        page_w = None
        if img_root is not None:
            page_w = try_load_page_width(page_name, img_root, exts=img_exts)

        # Build line ordering and text
        # lines_rows already has y0,y1 so we can order by y0
        lines_rows = sorted(lines_rows, key=lambda r: float(r["y0"]))
        trocr_map: Dict[int, str] = {}
        for r in trocr_rows:
            try:
                lid = int(r["line_id"])
            except Exception:
                continue
            trocr_map[lid] = normalize_text(r.get("text", ""))

        # Attach text and prepare inline buckets
        line_objs: List[dict] = []
        inline_by_line: Dict[int, List[Tuple[int, str]]] = {}  # line_id -> [(x, $latex$)]
        for ln in lines_rows:
            lid = int(ln["line_id"])
            ln2 = dict(ln)
            ln2["text"] = trocr_map.get(lid, "")
            line_objs.append(ln2)
            inline_by_line[lid] = []

        # Prepare display blocks
        display_items: List[Tuple[float, str]] = []

        # Assign math
        for mr in math_rows:
            kind = (mr.get("kind") or "").strip().lower()
            latex = (mr.get("latex") or "").strip()
            if not latex:
                continue

            x = float(mr.get("x", 0)); y = float(mr.get("y", 0))
            w = float(mr.get("w", 0)); h = float(mr.get("h", 0))
            cy = y + 0.5 * h

            if kind == "display":
                # Qwen output is already Markdown
                display_items.append((y, latex))
                continue

            # inline default
            inline = latex
            if not inline:
                continue
            lid = find_line_for_math(cy, line_objs)
            if lid is None:
                # fallback: treat as display-ish if we can't place it
                display_items.append((y, latex))

            else:
                inline_by_line[lid].append((int(x), inline))

        # Build page-level items in reading order:
        # - each text line as (y0, "text", content)
        # - each display math as (y, "math", block)
        items: List[Tuple[float, int, str]] = []  # (y, priority, content) priority: math before text at same y

        for ln in line_objs:
            lid = int(ln["line_id"])
            text = ln.get("text", "")
            segs = inline_by_line.get(lid, [])
            segs.sort(key=lambda t: t[0])
            merged_line = insert_inline_segments(text, segs, page_w=page_w)
            if merged_line:
                items.append((float(ln["y0"]), 1, merged_line))

        for y, block in display_items:
            block = block.strip()
            if block:
                items.append((float(y), 0, block))

        items.sort(key=lambda t: (t[0], t[1]))

        # Emit markdown
        md_lines: List[str] = []
        last_was_block = False
        for _, _, content in items:
            content = content.strip()
            if not content:
                continue

            is_block = ("\n" in content) or content.startswith("```")

            if is_block:
                if md_lines and md_lines[-1].strip() != "":
                    md_lines.append("")
                md_lines.append(content)
                md_lines.append("")
                last_was_block = True
            else:
                if last_was_block and md_lines and md_lines[-1].strip() != "":
                    md_lines.append("")
                md_lines.append(content)
                last_was_block = False

        # Clean trailing blank lines
        while md_lines and md_lines[-1].strip() == "":
            md_lines.pop()

        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"[OK] wrote {out_md}")

    print("done.")


if __name__ == "__main__":
    main()
