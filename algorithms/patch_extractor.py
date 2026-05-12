# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Tuple
import numpy as np

Rect = Tuple[int, int, int, int]  # x0,y0,x1,y1 in image coordinates


@dataclass
class PatchExtractConfig:
    rows: int = 6
    cols: int = 12
    sample_count: int = 100
    sample_center_area: float = 0.40
    card_crop_long: float = 0.01
    card_crop_short: float = 0.02
    random_seed: int = 0


def _normalize_rect(rect: Rect, width: int, height: int) -> Rect:
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    x0, x1 = sorted((max(0, min(width - 1, x0)), max(0, min(width, x1))))
    y0, y1 = sorted((max(0, min(height - 1, y0)), max(0, min(height, y1))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("标定矩形无效，请重新框选 Ref/Sample 区域。")
    return x0, y0, x1, y1


def _shrink_rect(rect: Rect, cfg: PatchExtractConfig) -> Rect:
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    dx = int(w * float(cfg.card_crop_long))
    dy = int(h * float(cfg.card_crop_short))
    return x0 + dx, y0 + dy, x1 - dx, y1 - dy


def _robust_mean_rgb(patch_rgb: np.ndarray, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if patch_rgb.size == 0:
        return np.zeros(3, dtype=np.float32)
    px = patch_rgb.reshape(-1, 3).astype(np.float32)
    if px.shape[0] >= sample_count:
        idx = rng.choice(px.shape[0], sample_count, replace=False)
    else:
        idx = rng.choice(px.shape[0], sample_count, replace=True)
    sel = px[idx]
    med = np.median(sel, axis=0)
    mad = np.median(np.abs(sel - med), axis=0) + 1e-6
    mask = np.all(np.abs(sel - med) <= 2.5 * mad, axis=1)
    sel = sel[mask]
    if sel.size == 0:
        sel = px[idx]
    return sel.mean(axis=0).astype(np.float32)


def extract_rgb_grid(image_rgb: np.ndarray, rect: Rect, cfg: PatchExtractConfig) -> np.ndarray:
    """从一个色卡矩形中提取 RGB 网格，返回 (3, rows, cols)，值域 0~255。"""
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"image_rgb 应为 (H,W,3)，实际为 {image_rgb.shape}")
    H, W = image_rgb.shape[:2]
    rect = _normalize_rect(rect, W, H)
    rect = _shrink_rect(rect, cfg)
    x0, y0, x1, y1 = _normalize_rect(rect, W, H)
    rows, cols = int(cfg.rows), int(cfg.cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("色卡行列数必须为正整数。")

    total_w, total_h = x1 - x0, y1 - y0
    cell_w = total_w / cols
    cell_h = total_h / rows
    side_ratio = np.sqrt(max(1e-6, min(1.0, float(cfg.sample_center_area))))
    margin_ratio = (1.0 - side_ratio) / 2.0
    rng = np.random.default_rng(int(cfg.random_seed))

    out = np.zeros((3, rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = x0 + (c + 1) * cell_w
            cy1 = y0 + (r + 1) * cell_h
            sx0 = int(round(cx0 + margin_ratio * cell_w))
            sy0 = int(round(cy0 + margin_ratio * cell_h))
            sx1 = int(round(cx1 - margin_ratio * cell_w))
            sy1 = int(round(cy1 - margin_ratio * cell_h))
            sx0, sy0 = max(0, sx0), max(0, sy0)
            sx1, sy1 = min(W, sx1), min(H, sy1)
            if sx1 <= sx0 or sy1 <= sy0:
                sx0, sy0 = int(round(cx0)), int(round(cy0))
                sx1, sy1 = int(round(cx1)), int(round(cy1))
            patch = image_rgb[sy0:sy1, sx0:sx1]
            out[:, r, c] = _robust_mean_rgb(patch, int(cfg.sample_count), rng)
    return out
