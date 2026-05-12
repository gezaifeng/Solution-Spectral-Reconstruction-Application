# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

Rect = Tuple[int, int, int, int]


@dataclass
class AutoDetectConfig:
    """自动识别上下色卡区域的参数。"""
    target_height: int = 512
    sobel_ksize: int = 3
    edge_thresh: int = 50
    det_brightness: float = 1.0
    det_contrast: float = 1.0
    det_saturation: float = 1.0
    det_gamma: float = 1.0
    det_use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_grid: int = 8
    det_auto_brighten: bool = True
    bright_target_median: float = 110.0
    bright_max_gain: float = 4.0
    bright_clip_low: float = 1.0
    bright_clip_high: float = 99.0
    bright_gamma_dark: float = 0.85
    bright_dark_thresh: float = 70.0
    min_cc_area: float = 0.005
    cc_topk: int = 10
    pair_center_constraint: bool = True
    pair_center_band: float = 0.30
    pair_min_dy: float = 0.12
    pair_max_area_ratio: float = 1.35
    pair_max_ar_ratio: float = 1.25


def _resize_keep_height(rgb: np.ndarray, target_h: int):
    h, w = rgb.shape[:2]
    if target_h <= 0 or h == target_h:
        return rgb.copy(), (w, h, w, h)
    scale = float(target_h) / float(h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return resized, (w, h, nw, nh)


def _to_uint8_gray(img) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    if gray.dtype == np.uint8:
        return gray
    if gray.dtype == np.uint16:
        return (gray / 257).astype(np.uint8)
    g = gray.astype(np.float32)
    g = g - g.min()
    mx = g.max()
    if mx > 1e-6:
        g = g / mx
    return (g * 255.0).clip(0, 255).astype(np.uint8)


def _apply_det_tuning_bgr(bgr: np.ndarray, cfg: AutoDetectConfig) -> np.ndarray:
    img = bgr
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * float(cfg.det_saturation), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * float(cfg.det_brightness), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if abs(float(cfg.det_contrast) - 1.0) > 1e-6:
        img = cv2.convertScaleAbs(img, alpha=float(cfg.det_contrast), beta=0)

    if abs(float(cfg.det_gamma) - 1.0) > 1e-6:
        g = float(np.clip(cfg.det_gamma, 0.05, 5.0))
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255.0 for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, lut)
    return img


def _apply_clahe_gray8(gray8: np.ndarray, cfg: AutoDetectConfig) -> np.ndarray:
    if not bool(cfg.det_use_clahe):
        return gray8
    grid = int(max(2, min(32, cfg.clahe_grid)))
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip_limit), tileGridSize=(grid, grid))
    return clahe.apply(gray8)


def _auto_brighten_gray(gray8: np.ndarray, cfg: AutoDetectConfig) -> np.ndarray:
    if not bool(cfg.det_auto_brighten):
        return gray8
    g = gray8.astype(np.float32)
    lo = np.percentile(g, float(cfg.bright_clip_low))
    hi = np.percentile(g, float(cfg.bright_clip_high))
    if hi <= lo + 1e-6:
        return gray8
    g = np.clip(g, lo, hi)
    g = (g - lo) * (255.0 / (hi - lo + 1e-6))
    med = float(np.median(g))
    if med < 1e-6:
        med = 1.0
    gain = float(np.clip(float(cfg.bright_target_median) / med, 1.0, float(cfg.bright_max_gain)))
    g = np.clip(g * gain, 0, 255)
    if med < float(cfg.bright_dark_thresh):
        inv = 1.0 / max(1e-6, float(cfg.bright_gamma_dark))
        g = 255.0 * ((g / 255.0) ** inv)
    return np.clip(g, 0, 255).astype(np.uint8)


def _box_to_rect(box: np.ndarray) -> Rect:
    xs = box[:, 0]
    ys = box[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def detect_regions_pair_bgr(det_bgr: np.ndarray, cfg: AutoDetectConfig):
    det_bgr2 = _apply_det_tuning_bgr(det_bgr, cfg)
    gray8 = _to_uint8_gray(det_bgr2)
    if gray8 is None:
        return {"gray": None}, None, None
    gray8 = _apply_clahe_gray8(gray8, cfg)
    gray8 = _auto_brighten_gray(gray8, cfg)

    k = int(cfg.sobel_ksize)
    if k not in (1, 3, 5, 7):
        k = 3
    gx = cv2.Sobel(gray8, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray8, cv2.CV_64F, 0, 1, ksize=k)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_u8 = (255.0 * mag / (mag.max() + 1e-8)).astype(np.uint8)
    _, binary = cv2.threshold(mag_u8, int(cfg.edge_thresh), 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dbg = {"mag": mag_u8, "binary": binary, "gray": gray8}
    if num_labels < 3:
        return dbg, None, None

    def box_aabb(box):
        xs = box[:, 0]; ys = box[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def aabb_intersection_area(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0
        return (ix2 - ix1) * (iy2 - iy1)

    def aabb_contains(outer, inner, margin=0):
        ox1, oy1, ox2, oy2 = outer
        ix1, iy1, ix2, iy2 = inner
        return (ox1 <= ix1 - margin and oy1 <= iy1 - margin and ox2 >= ix2 + margin and oy2 >= iy2 + margin)

    def rotated_rect_intersect(r1, r2):
        inter_type, _ = cv2.rotatedRectangleIntersection(r1, r2)
        return inter_type != 0

    def poly_contains(poly_outer, poly_inner):
        for p in poly_inner:
            if cv2.pointPolygonTest(poly_outer.astype(np.float32), (float(p[0]), float(p[1])), False) < 0:
                return False
        return True

    H, W = gray8.shape[:2]
    min_area = int(float(cfg.min_cc_area) * (H * W))
    areas = stats[1:, cv2.CC_STAT_AREA]
    idxs = np.argsort(areas)[::-1] + 1
    candidates = []
    for lab in idxs[:int(cfg.cc_topk)]:
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 10:
            continue
        candidates.append({
            "lab": lab, "area": float(stats[lab, cv2.CC_STAT_AREA]), "rect": rect,
            "box": box, "aabb": box_aabb(box), "cy": float(cy), "cx": float(cx)
        })

    if len(candidates) < 2:
        return dbg, None, None

    best = None
    best_score = -1e18
    center_x = W * 0.5
    center_band = float(cfg.pair_center_band) * W
    min_dy = float(cfg.pair_min_dy) * H
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            A = candidates[i]; B = candidates[j]
            inter = aabb_intersection_area(A["aabb"], B["aabb"])
            if inter > 0 and rotated_rect_intersect(A["rect"], B["rect"]):
                continue
            if aabb_contains(A["aabb"], B["aabb"], margin=2) and poly_contains(A["box"], B["box"]):
                continue
            if aabb_contains(B["aabb"], A["aabb"], margin=2) and poly_contains(B["box"], A["box"]):
                continue
            if bool(cfg.pair_center_constraint):
                if abs(A["cx"] - center_x) > center_band or abs(B["cx"] - center_x) > center_band:
                    continue
            if abs(A["cy"] - B["cy"]) < min_dy:
                continue
            (_, _), (aw, ah), _ = A["rect"]
            (_, _), (bw, bh), _ = B["rect"]
            aw, ah = max(aw, 1e-6), max(ah, 1e-6)
            bw, bh = max(bw, 1e-6), max(bh, 1e-6)
            area_ratio = max(aw * ah, bw * bh) / max(min(aw * ah, bw * bh), 1e-6)
            a_ar = max(aw, ah) / min(aw, ah)
            b_ar = max(bw, bh) / min(bw, bh)
            if area_ratio > float(cfg.pair_max_area_ratio):
                continue
            if max(a_ar, b_ar) / max(min(a_ar, b_ar), 1e-6) > float(cfg.pair_max_ar_ratio):
                continue
            score = (A["area"] + B["area"])
            score -= 0.5 * (abs(A["cx"] - center_x) + abs(B["cx"] - center_x))
            score -= 3.0 * max(0.0, (min_dy - abs(A["cy"] - B["cy"])))
            if score > best_score:
                best_score = score
                best = (A, B)
    if best is None:
        return dbg, None, None
    A, B = best
    ref_box, sample_box = (A["box"], B["box"]) if A["cy"] < B["cy"] else (B["box"], A["box"])
    vis = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, [ref_box], True, (0, 255, 0), 2)
    cv2.polylines(vis, [sample_box], True, (255, 0, 0), 2)
    dbg["vis"] = vis
    return dbg, ref_box.astype(np.int32), sample_box.astype(np.int32)


def auto_detect_rects(image_rgb: np.ndarray, cfg: Optional[AutoDetectConfig] = None):
    """输入 RGB 图像，返回 ref_rect, sample_rect, debug。rect 为原图坐标中的 x0,y0,x1,y1。"""
    if cfg is None:
        cfg = AutoDetectConfig()
    small_rgb, (ow, oh, nw, nh) = _resize_keep_height(image_rgb, int(cfg.target_height))
    small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
    debug, ref_box_s, sample_box_s = detect_regions_pair_bgr(small_bgr, cfg)
    if ref_box_s is None or sample_box_s is None:
        return None, None, debug
    scale_x, scale_y = ow / float(nw), oh / float(nh)
    ref_box = np.array([[int(round(x * scale_x)), int(round(y * scale_y))] for x, y in ref_box_s], dtype=np.int32)
    sample_box = np.array([[int(round(x * scale_x)), int(round(y * scale_y))] for x, y in sample_box_s], dtype=np.int32)
    ref_rect = _box_to_rect(ref_box)
    sample_rect = _box_to_rect(sample_box)
    return ref_rect, sample_rect, debug
