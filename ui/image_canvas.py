# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple, Callable
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtWidgets import QWidget

Rect = Tuple[int, int, int, int]


class ImageCanvas(QWidget):
    rectChanged = Signal(str, tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.image_rgb: Optional[np.ndarray] = None
        self.qimg: Optional[QImage] = None
        self.mode: Optional[str] = None  # 'ref' or 'sample'
        self.rect_ref: Optional[Rect] = None
        self.rect_sample: Optional[Rect] = None
        self._drag_start_img: Optional[QPointF] = None
        self._drag_current_img: Optional[QPointF] = None
        self._display_rect = QRectF()
        self.rows = 6
        self.cols = 12
        self.crop_long = 0.01
        self.crop_short = 0.02

    def set_image(self, image_rgb: np.ndarray):
        self.image_rgb = image_rgb
        h, w = image_rgb.shape[:2]
        arr = np.ascontiguousarray(image_rgb)
        self.qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.update()

    def set_rects(self, ref: Optional[Rect], sample: Optional[Rect]):
        self.rect_ref = ref
        self.rect_sample = sample
        self.update()

    def set_grid(self, rows: int, cols: int):
        self.rows = max(1, int(rows))
        self.cols = max(1, int(cols))
        self.update()

    def set_crop(self, crop_long: float, crop_short: float):
        self.crop_long = max(0.0, min(0.45, float(crop_long)))
        self.crop_short = max(0.0, min(0.45, float(crop_short)))
        self.update()

    def set_mode(self, mode: Optional[str]):
        self.mode = mode
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)
        self.update()

    def clear_rects(self):
        self.rect_ref = None
        self.rect_sample = None
        self.update()

    def _calc_display_rect(self) -> QRectF:
        if self.qimg is None:
            return QRectF(0, 0, self.width(), self.height())
        iw, ih = self.qimg.width(), self.qimg.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / iw, wh / ih)
        dw, dh = iw * scale, ih * scale
        x = (ww - dw) / 2
        y = (wh - dh) / 2
        return QRectF(x, y, dw, dh)

    def _widget_to_image(self, p) -> Optional[QPointF]:
        if self.qimg is None:
            return None
        r = self._display_rect
        if not r.contains(QPointF(p)):
            return None
        x = (p.x() - r.x()) / r.width() * self.qimg.width()
        y = (p.y() - r.y()) / r.height() * self.qimg.height()
        x = max(0.0, min(float(self.qimg.width() - 1), x))
        y = max(0.0, min(float(self.qimg.height() - 1), y))
        return QPointF(x, y)

    def _image_to_widget_rect(self, rect: Rect) -> QRectF:
        x0, y0, x1, y1 = rect
        r = self._display_rect
        iw, ih = self.qimg.width(), self.qimg.height()
        wx0 = r.x() + x0 / iw * r.width()
        wy0 = r.y() + y0 / ih * r.height()
        wx1 = r.x() + x1 / iw * r.width()
        wy1 = r.y() + y1 / ih * r.height()
        return QRectF(QPointF(wx0, wy0), QPointF(wx1, wy1)).normalized()

    def _points_to_rect(self, a: QPointF, b: QPointF) -> Rect:
        x0, x1 = sorted((int(round(a.x())), int(round(b.x()))))
        y0, y1 = sorted((int(round(a.y())), int(round(b.y()))))
        return x0, y0, x1, y1

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.mode in ("ref", "sample") and self.qimg is not None:
            p = self._widget_to_image(event.position())
            if p is not None:
                self._drag_start_img = p
                self._drag_current_img = p
                self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start_img is not None:
            p = self._widget_to_image(event.position())
            if p is not None:
                self._drag_current_img = p
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_start_img is not None and self._drag_current_img is not None:
            rect = self._points_to_rect(self._drag_start_img, self._drag_current_img)
            if abs(rect[2] - rect[0]) >= 8 and abs(rect[3] - rect[1]) >= 8:
                if self.mode == "ref":
                    self.rect_ref = rect
                    self.rectChanged.emit("ref", rect)
                elif self.mode == "sample":
                    self.rect_sample = rect
                    self.rectChanged.emit("sample", rect)
            self._drag_start_img = None
            self._drag_current_img = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f5f6f8"))
        self._display_rect = self._calc_display_rect()
        if self.qimg is None:
            painter.setPen(QColor("#7b7f87"))
            painter.setFont(QFont("Microsoft YaHei", 16))
            painter.drawText(self.rect(), Qt.AlignCenter, "请先导入图片")
            return
        painter.drawImage(self._display_rect, self.qimg)
        painter.setPen(QPen(QColor("#d4d7dd"), 1))
        painter.drawRoundedRect(self._display_rect, 10, 10)

        def draw_labeled_rect(rect: Optional[Rect], color: QColor, label: str):
            if rect is None:
                return
            wr = self._image_to_widget_rect(rect)
            border = QColor(color.red(), color.green(), color.blue(), 245)
            # 只画外边缘，不用大面积色块覆盖图像，方便检查自动识别是否准确
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(border, 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawRect(wr)

            painter.setBrush(QBrush(border))
            painter.setPen(Qt.white)
            painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
            tag = QRectF(wr.x(), max(0, wr.y() - 24), 92, 22)
            painter.drawRoundedRect(tag, 6, 6)
            painter.drawText(tag, Qt.AlignCenter, label)

            # inner crop region used for extraction, to avoid black border
            dx = wr.width() * self.crop_long
            dy = wr.height() * self.crop_short
            inner = QRectF(wr.x() + dx, wr.y() + dy, max(1.0, wr.width() - 2 * dx), max(1.0, wr.height() - 2 * dy))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 220), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawRect(inner)

            # 内部数学分割：细线显示实际采样网格，不遮挡图像内容
            painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 130), 0.8, Qt.SolidLine))
            for c in range(1, self.cols):
                x = inner.x() + inner.width() * c / self.cols
                painter.drawLine(x, inner.y(), x, inner.y() + inner.height())
            for rr in range(1, self.rows):
                y = inner.y() + inner.height() * rr / self.rows
                painter.drawLine(inner.x(), y, inner.x() + inner.width(), y)

        draw_labeled_rect(self.rect_ref, QColor("#2f80ed"), "Ref 参考")
        draw_labeled_rect(self.rect_sample, QColor("#27ae60"), "Sample 样本")

        if self._drag_start_img is not None and self._drag_current_img is not None:
            temp = self._points_to_rect(self._drag_start_img, self._drag_current_img)
            wr = self._image_to_widget_rect(temp)
            color = QColor("#2f80ed") if self.mode == "ref" else QColor("#27ae60")
            painter.setPen(QPen(color, 2, Qt.DashLine))
            painter.drawRect(wr)
