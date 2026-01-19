import os
import sys
import csv
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QAction, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QCheckBox,
    QMessageBox,
)


# ----------------------------
# Utils
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    """Convert PIL RGB image to QImage safely."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)  # H, W, 3 (RGB)
    h, w, ch = arr.shape
    bytes_per_line = ch * w
    # copy() is important because arr memory can be freed
    return QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


# ----------------------------
# Data
# ----------------------------

@dataclass
class Patch:
    x: int
    y: int
    w: int
    h: int
    label: int = 0  # 0=text, 1=math


# ----------------------------
# Canvas
# ----------------------------

class Canvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.img: Optional[Image.Image] = None
        self.qimg: Optional[QImage] = None
        self.pixmap: Optional[QPixmap] = None
        self.W: int = 0
        self.H: int = 0

        self.patch_size: int = 96
        self.patches: List[Patch] = []

        self.show_grid: bool = True
        self.math_alpha: int = 90

        # view transform
        self.scale: float = 1.0
        self.offset: QPointF = QPointF(0.0, 0.0)

        # interactions
        self._panning: bool = False
        self._last_mouse: QPointF = QPointF(0.0, 0.0)

        self._painting: bool = False
        self._paint_math: bool = True  # True => label math; False => label text
        self._last_patch_idx: Optional[int] = None

    def has_image(self) -> bool:
        return self.qimg is not None

    def set_image(self, path: str) -> None:
        self.img = Image.open(path).convert("RGB")
        self.qimg = pil_to_qimage(self.img)
        self.pixmap = QPixmap.fromImage(self.qimg)
        self.W, self.H = self.img.size

        # reset view
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)

        self.rebuild_patches()
        self.update()

    def set_patch_size(self, s: int) -> None:
        self.patch_size = max(16, int(s))
        if self.has_image():
            self.rebuild_patches()
            self.update()

    def rebuild_patches(self) -> None:
        self.patches = []
        if not self.has_image():
            return
        ps = self.patch_size
        for y in range(0, self.H, ps):
            for x in range(0, self.W, ps):
                w = min(ps, self.W - x)
                h = min(ps, self.H - y)
                self.patches.append(Patch(x=x, y=y, w=w, h=h, label=0))

    def widget_to_image(self, p: QPointF) -> QPointF:
        # (widget - offset) / scale
        return QPointF((p.x() - self.offset.x()) / self.scale,
                       (p.y() - self.offset.y()) / self.scale)

    def patch_index_at(self, ip: QPointF) -> Optional[int]:
        if not self.has_image():
            return None
        x, y = int(ip.x()), int(ip.y())
        if x < 0 or y < 0 or x >= self.W or y >= self.H:
            return None

        ps = self.patch_size
        cols = (self.W + ps - 1) // ps
        col = x // ps
        row = y // ps
        idx = row * cols + col

        if 0 <= idx < len(self.patches):
            return idx
        return None

    def _paint_at(self, widget_pos: QPointF) -> None:
        ip = self.widget_to_image(widget_pos)
        idx = self.patch_index_at(ip)
        if idx is None:
            return
        if idx == self._last_patch_idx:
            return

        self._last_patch_idx = idx
        self.patches[idx].label = 1 if self._paint_math else 0

    def export_mask(self) -> np.ndarray:
        """Binary mask: 255 where math, else 0."""
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for p in self.patches:
            if p.label == 1:
                mask[p.y:p.y + p.h, p.x:p.x + p.w] = 255
        return mask

    def crop_patch(self, p: Patch) -> Image.Image:
        assert self.img is not None
        return self.img.crop((p.x, p.y, p.x + p.w, p.y + p.h))

    # ---- Qt events ----

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))

        if not self.has_image() or self.pixmap is None:
            painter.setPen(QColor(220, 220, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open an image to label.")
            return

        painter.save()
        painter.translate(self.offset)
        painter.scale(self.scale, self.scale)

        # base image
        painter.drawPixmap(0, 0, self.pixmap)

        # overlay math patches
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 0, 0, self.math_alpha))
        for p in self.patches:
            if p.label == 1:
                painter.drawRect(p.x, p.y, p.w, p.h)

        # grid
        if self.show_grid:
            pen = QPen(QColor(0, 255, 255, 110))
            pen.setWidthF(1.0 / max(self.scale, 1e-6))
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for p in self.patches:
                painter.drawRect(p.x, p.y, p.w, p.h)

        painter.restore()

        # HUD
        n_math = sum(1 for p in self.patches if p.label == 1)
        painter.setPen(QColor(240, 240, 240))
        painter.drawText(
            10, 20,
            f"Zoom {self.scale:.2f} | Patch {self.patch_size}px | Math patches: {n_math}"
        )

    def wheelEvent(self, e: QWheelEvent) -> None:
        if not self.has_image():
            return

        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)

        old_scale = self.scale
        new_scale = max(0.1, min(20.0, old_scale * factor))

        cursor = QPointF(e.position())
        img_before = self.widget_to_image(cursor)

        self.scale = new_scale

        # Keep cursor stable by adjusting offset
        cursor_after = QPointF(img_before.x() * self.scale + self.offset.x(),
                               img_before.y() * self.scale + self.offset.y())
        self.offset += (cursor - cursor_after)

        self.update()

    def mousePressEvent(self, e) -> None:
        if not self.has_image():
            return

        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_mouse = QPointF(e.position())
            return

        if e.button() == Qt.MouseButton.LeftButton:
            self._painting = True
            self._paint_math = True
            self._last_patch_idx = None
            self._paint_at(QPointF(e.position()))
            self.update()
            return

        if e.button() == Qt.MouseButton.RightButton:
            self._painting = True
            self._paint_math = False
            self._last_patch_idx = None
            self._paint_at(QPointF(e.position()))
            self.update()
            return

    def mouseMoveEvent(self, e) -> None:
        if not self.has_image():
            return

        if self._panning:
            cur = QPointF(e.position())
            self.offset += (cur - self._last_mouse)
            self._last_mouse = cur
            self.update()
            return

        if self._painting:
            self._paint_at(QPointF(e.position()))
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            return

        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self._painting = False
            self._last_patch_idx = None


# ----------------------------
# Main window
# ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Full-Page Patch Painter → Patch Dataset for SmallCNN")
        self.resize(1200, 800)

        self.canvas = Canvas()
        self.current_path: Optional[str] = None

        # Controls
        self.btn_open = QPushButton("Open Image")
        self.btn_export = QPushButton("Export dataset")
        self.btn_export.setEnabled(False)

        self.spin_patch = QSpinBox()
        self.spin_patch.setRange(32, 512)
        self.spin_patch.setValue(96)

        self.chk_grid = QCheckBox("Show grid")
        self.chk_grid.setChecked(True)

        self.info = QLabel(
            "Controls:\n"
            "  Left-drag = label MATH\n"
            "  Right-drag = label TEXT (erase)\n"
            "  Middle-drag = pan, Wheel = zoom\n"
            "Shortcut: G toggles grid\n\n"
            "Export writes: mask.png + patches.csv + dataset/(math,text)/patches"
        )
        self.info.setStyleSheet("color:#ddd;")

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(QLabel("Patch size:"))
        top.addWidget(self.spin_patch)
        top.addWidget(self.chk_grid)
        top.addStretch(1)
        top.addWidget(self.btn_export)

        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.info)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)

        # Wiring
        self.btn_open.clicked.connect(self.open_image)
        self.btn_export.clicked.connect(self.export_dataset)
        self.spin_patch.valueChanged.connect(self.on_patch_size_changed)
        self.chk_grid.stateChanged.connect(self.on_grid_toggled)

        # Shortcut G: toggle grid
        act_toggle_grid = QAction(self)
        act_toggle_grid.setShortcut("G")
        act_toggle_grid.triggered.connect(self.toggle_grid)
        self.addAction(act_toggle_grid)

    def toggle_grid(self):
        self.chk_grid.setChecked(not self.chk_grid.isChecked())

    def on_grid_toggled(self):
        self.canvas.show_grid = self.chk_grid.isChecked()
        self.canvas.update()

    def on_patch_size_changed(self, v: int):
        self.canvas.set_patch_size(int(v))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self.canvas.set_image(path)
            self.current_path = path
            self.btn_export.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open image:\n{e}")

    def export_dataset(self):
        if not self.canvas.has_image() or self.current_path is None:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        ensure_dir(out_dir)

        base = os.path.splitext(os.path.basename(self.current_path))[0]

        # 1) mask
        mask = self.canvas.export_mask()
        mask_path = os.path.join(out_dir, f"{base}_math_mask.png")
        cv2.imwrite(mask_path, mask)

        # 2) dataset dirs
        ds_dir = os.path.join(out_dir, "dataset")
        math_dir = os.path.join(ds_dir, "math")
        text_dir = os.path.join(ds_dir, "text")
        ensure_dir(math_dir)
        ensure_dir(text_dir)

        # 3) CSV + patch crops
        csv_path = os.path.join(out_dir, f"{base}_patches.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["img_path", "x", "y", "w", "h", "patch_size", "label", "patch_file"])

            for p in self.canvas.patches:
                label = "math" if p.label == 1 else "text"
                patch_img = self.canvas.crop_patch(p)

                name = f"{base}__{p.x}_{p.y}_{self.canvas.patch_size}__{label}.png"
                out_path = os.path.join(math_dir if label == "math" else text_dir, name)
                patch_img.save(out_path)

                w.writerow([self.current_path, p.x, p.y, p.w, p.h, self.canvas.patch_size, label, out_path])

        QMessageBox.information(
            self,
            "Export complete",
            f"Saved:\n- {mask_path}\n- {csv_path}\n- {ds_dir}{os.sep}(math,text){os.sep}"
        )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
