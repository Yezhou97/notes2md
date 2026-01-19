import argparse, csv
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import cv2


def read_pred_csv(csv_path: Path):
    patches: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
    img_path = None
    grid = None

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if img_path is None:
                img_path = row["img_path"]
            x = int(row["x"]); y = int(row["y"])
            w = int(row["w"]); h = int(row["h"])
            grid = int(row["grid"])
            p = float(row["p_math"])
            patches[(x, y)] = (w, h, p)

    if img_path is None or grid is None or not patches:
        raise RuntimeError(f"Empty/invalid CSV: {csv_path}")

    xs = [x for x,_ in patches.keys()]
    ys = [y for _,y in patches.keys()]
    R = max(ys)//grid + 1
    C = max(xs)//grid + 1

    prob = np.zeros((R, C), dtype=np.float32)
    wh = np.zeros((R, C, 2), dtype=np.int32)

    for (x, y), (w, h, p) in patches.items():
        rr = y // grid
        cc = x // grid
        prob[rr, cc] = p
        wh[rr, cc, 0] = w
        wh[rr, cc, 1] = h

    return Path(img_path), int(grid), prob, wh


def grid_to_pixel_box(r1, r2, c1, c2, grid, H, W, pad_x=0, pad_y=0):
    # r2/c2 are exclusive
    x1 = c1 * grid
    y1 = r1 * grid
    x2 = c2 * grid
    y2 = r2 * grid
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x); y2 = min(H, y2 + pad_y)
    return (x1, y1, x2 - x1, y2 - y1)


def horiz_close(mask_rc: np.ndarray, kx: int):
    if kx <= 1:
        return mask_rc
    k = np.ones((1, kx), np.uint8)
    m = (mask_rc.astype(np.uint8) * 255)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m > 0


def connected_components_grid(mask_rc: np.ndarray):
    m = mask_rc.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # stats[i] in grid cells: x,y,w,h,area
    return num, labels, stats


def component_filter_display(prob, thr, thr_peak, min_cells, min_fill, min_mean, max_bbox_cells, close_x):
    base = (prob >= thr)
    base = horiz_close(base, close_x)

    num, labels, stats = connected_components_grid(base)
    keep = np.zeros_like(base, dtype=bool)

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        bbox_area = int(w * h)
        if area < min_cells:
            continue
        if bbox_area > max_bbox_cells:
            continue

        comp = (labels == i)
        maxp = float(prob[comp].max())
        meanp = float(prob[comp].mean())
        fill = float(area) / float(max(1, bbox_area))

        if maxp < thr_peak:
            continue
        if not (fill >= min_fill or meanp >= min_mean):
            continue

        keep[comp] = True

    return keep


def inline_runs_from_prob(prob, thr_inline, thr_peak_inline, min_run, max_height_rows, close_x):
    """
    Build inline boxes by scanning row bands:
    - take candidate cells prob>=thr_inline
    - optional horizontal close within row
    - find contiguous runs per row; keep run if (max prob in run)>=thr_peak_inline and length>=min_run
    - optionally allow 2-row boxes (max_height_rows=2) by OR-ing adjacent rows and then re-running.
    """
    R, C = prob.shape
    cand = (prob >= thr_inline)
    cand = horiz_close(cand, close_x)

    boxes = []

    def row_runs(mask_row, prob_row, r, height_rows=1):
        c = 0
        while c < C:
            if not mask_row[c]:
                c += 1
                continue
            c1 = c
            while c < C and mask_row[c]:
                c += 1
            c2 = c  # exclusive
            if (c2 - c1) >= min_run:
                peak = float(prob_row[c1:c2].max())
                if peak >= thr_peak_inline:
                    boxes.append((r, r + height_rows, c1, c2))
        return

    # 1-row inline boxes
    for r in range(R):
        row_runs(cand[r, :], prob[r, :], r, height_rows=1)

    if max_height_rows >= 2:
        # 2-row inline boxes: combine r and r+1, but still keep Y bounded
        for r in range(R - 1):
            comb = cand[r, :] | cand[r + 1, :]
            comb_prob = np.maximum(prob[r, :], prob[r + 1, :])
            row_runs(comb, comb_prob, r, height_rows=2)

    # Deduplicate (many overlaps). Keep the tighter (smaller height) first.
    boxes = sorted(set(boxes), key=lambda b: (b[1]-b[0], b[0], b[2]))
    return boxes


def merge_inline_boxes_horiz(boxes, merge_gap_cols=1):
    """
    Merge inline boxes only if they are on same (r1,r2) band and close in X.
    boxes are (r1,r2,c1,c2)
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[0], b[1], b[2]))
    out = []
    cur = list(boxes[0])
    for r1,r2,c1,c2 in boxes[1:]:
        if r1 == cur[0] and r2 == cur[1] and c1 <= cur[3] + merge_gap_cols:
            cur[3] = max(cur[3], c2)
        else:
            out.append(tuple(cur))
            cur = [r1,r2,c1,c2]
    out.append(tuple(cur))
    return out


def draw_boxes(bgr, boxes, color, thickness=2):
    out = bgr.copy()
    for x,y,w,h in boxes:
        cv2.rectangle(out, (x,y), (x+w,y+h), color, thickness)
    return out

Box = Tuple[float, float, float, float]  # (x,y,w,h)

def box_area(b: Box) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])

def inter_area(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax, ay, ax + aw, ay + ah
    bx0, by0, bx1, by1 = bx, by, bx + bw, by + bh
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

def contains(a: Box, b: Box, thr: float = 0.85) -> bool:
    """Return True if 'a' contains >=thr fraction of 'b' area."""
    bA = box_area(b)
    if bA <= 1e-9:
        return False
    return inter_area(a, b) / bA >= thr

def median_h(boxes: List[Box]) -> float:
    hs = [h for (_, _, _, h) in boxes if h > 1]
    return float(np.median(hs)) if hs else 0.0

def filter_display_boxes(
    disp_boxes: List[Box],
    inline_boxes: List[Box],
    min_score: int = 2,
    tall_ratio: float = 1.6,
    contain_thr: float = 0.85,
) -> List[Box]:
    """
    Score-based display filter.
    """
    med_inline_h = median_h(inline_boxes)
    med_inline_w = float(np.median([w for (_,_,w,_) in inline_boxes])) if inline_boxes else 0.0
    med_inline_area = float(np.median([box_area(b) for b in inline_boxes])) if inline_boxes else 0.0

    kept = []
    for d in disp_boxes:
        score = 0
        dx, dy, dw, dh = d
        dA = box_area(d)

        # 1) height signal
        if med_inline_h > 0 and dh >= tall_ratio * med_inline_h:
            score += 1

        # 2) containment signal
        k = sum(1 for b in inline_boxes if contains(d, b, thr=contain_thr))
        if k <= 1:
            score += 1

        # 3) width signal (display math tends to be wide)
        if med_inline_w > 0 and dw >= 1.6 * med_inline_w:
            score += 1

        # 4) area signal
        if med_inline_area > 0 and dA >= 2.5 * med_inline_area:
            score += 1

        if score >= min_score:
            kept.append(d)

    return kept

def boxes_to_mask(H: int, W: int, boxes: List[Box], pad_x: int = 8, pad_y: int = 6) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in boxes:
        x0 = max(0, int(x - pad_x))
        y0 = max(0, int(y - pad_y))
        x1 = min(W, int(x + w + pad_x))
        y1 = min(H, int(y + h + pad_y))
        mask[y0:y1, x0:x1] = 255
    return mask

def make_text_only(bgr: np.ndarray, math_boxes: List[Box], pad_x: int = 8, pad_y: int = 6, mode: str = "inpaint") -> np.ndarray:
    """
    mode: 'inpaint' (recommended) or 'white'
    """
    H, W = bgr.shape[:2]
    mask = boxes_to_mask(H, W, math_boxes, pad_x=pad_x, pad_y=pad_y)

    if mode == "white":
        out = bgr.copy()
        out[mask > 0] = (255, 255, 255)
        return out

    # inpaint tends to preserve paper background better
    return cv2.inpaint(bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="one *_pred_patches.csv OR a directory")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--glob", default="*_pred_patches.csv")

    # inline boxes
    ap.add_argument("--thr_inline", type=float, default=0.50)
    ap.add_argument("--thr_peak_inline", type=float, default=0.68)
    ap.add_argument("--min_run", type=int, default=2, help="min contiguous cells in a run")
    ap.add_argument("--max_height_rows", type=int, default=1, help="1 or 2")
    ap.add_argument("--inline_close_x", type=int, default=3)
    ap.add_argument("--inline_merge_gap_cols", type=int, default=1)

    # pixel padding for cropping
    ap.add_argument("--pad_x", type=int, default=10)
    ap.add_argument("--pad_y", type=int, default=2)

    # filter messy display boxes using inline boxes

    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = [in_path] if in_path.is_file() else sorted(in_path.rglob(args.glob))
    if not csvs:
        raise SystemExit("no csvs found")

    for csv_path in csvs:
        img_path, grid, prob, wh = read_pred_csv(csv_path)
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[FAIL] {csv_path}: cannot read image {img_path}")
            continue
        H, W = bgr.shape[:2]

        # # Display mask -> display boxes in pixel coords
        # disp_mask = component_filter_display(
        #     prob, args.thr, args.thr_peak, args.min_cells, args.min_fill, args.min_mean,
        #     args.max_bbox_cells, args.display_close_x
        # )
        # # connected components in grid for display boxes
        # num, labels, stats = connected_components_grid(disp_mask)
        # disp_boxes = []
        # for i in range(1, num):
        #     x,y,w,h,area = stats[i]
        #     disp_boxes.append(grid_to_pixel_box(y, y+h, x, x+w, grid, H, W, pad_x=args.pad_x, pad_y=args.pad_y))

        # Inline boxes from row runs
        inline_grid_boxes = inline_runs_from_prob(
            prob, args.thr_inline, args.thr_peak_inline, args.min_run,
            args.max_height_rows, args.inline_close_x
        )
        inline_grid_boxes = merge_inline_boxes_horiz(inline_grid_boxes, merge_gap_cols=args.inline_merge_gap_cols)
        inline_boxes = [grid_to_pixel_box(r1, r2, c1, c2, grid, H, W, pad_x=args.pad_x, pad_y=args.pad_y)
                        for (r1,r2,c1,c2) in inline_grid_boxes]

                # --- NEW: filter messy display boxes using inline boxes ---
        # disp_boxes_raw = disp_boxes
        # disp_boxes = filter_display_boxes(
        #     disp_boxes=disp_boxes_raw,
        #     inline_boxes=inline_boxes,
        #     min_score=2,
        #     tall_ratio=1.6,   # much softer than 2.2
        # )


        base = csv_path.stem.replace("_pred_patches", "")
        out_dir = out_root / base
        out_dir.mkdir(parents=True, exist_ok=True)

        # write CSVs
        # with open(out_dir / f"{base}_math_boxes_display.csv", "w", newline="", encoding="utf-8") as f:
        #     w = csv.writer(f); w.writerow(["x","y","w","h"])
        #     for b in disp_boxes: w.writerow(list(b))

        with open(out_dir / f"{base}_math_boxes_inline.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["x","y","w","h"])
            for b in inline_boxes: w.writerow(list(b))
        

        # overlay: display = red, inline = blue
        ov = bgr.copy()
        # ov = draw_boxes(ov, disp_boxes, (0,0,255), 2)
        ov = draw_boxes(ov, inline_boxes, (255,0,0), 1)
        cv2.imwrite(str(out_dir / f"{base}_overlay_boxes.png"), ov)

        # generate text-only image by removing math boxes
        math_boxes_all = inline_boxes  # both are (x,y,w,h)

        text_only = make_text_only(bgr, math_boxes_all, pad_x=args.pad_x, pad_y=args.pad_y, mode="white")
        cv2.imwrite(str(out_dir / f"{base}_text_only.png"), text_only)


    print("done.")


if __name__ == "__main__":
    main()
