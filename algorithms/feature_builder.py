# -*- coding: utf-8 -*-
import numpy as np


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def sanitize_feature(x_chw: np.ndarray, clip_value: float = 8.0) -> np.ndarray:
    x = np.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_value is not None:
        x = np.clip(x, -float(clip_value), float(clip_value))
    return x.astype(np.float32)


def per_image_channel_zscore(x_chw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x_chw.astype(np.float32)
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    std_safe = np.where(std < 1e-4, 1.0, std)
    return ((x - mean) / (std_safe + eps)).astype(np.float32)


def build_log_ratio_feature(ref_rgb_346: np.ndarray, sample_rgb_346: np.ndarray,
                            clip_value: float = 8.0, use_zscore: bool = True,
                            eps: float = 1e-6) -> tuple[np.ndarray, dict]:
    """构建与训练脚本一致的 log_ratio 输入特征。

    输入 ref/sample 均为 (3, rows, cols)，0~255 RGB。
    输出 X 为 (3, rows, cols)，float32。
    """
    ref = ref_rgb_346.astype(np.float32) / 255.0
    sam = sample_rgb_346.astype(np.float32) / 255.0
    ref_lin = srgb_to_linear(ref)
    sam_lin = srgb_to_linear(sam)
    ratio = sam_lin / np.clip(ref_lin, eps, None)
    log_ratio = np.log(np.clip(sam_lin, eps, None)) - np.log(np.clip(ref_lin, eps, None))
    X_raw = log_ratio.astype(np.float32)
    X = sanitize_feature(X_raw, clip_value=clip_value)
    if use_zscore:
        X = per_image_channel_zscore(X)
    extras = {
        "ref_lin": ref_lin.astype(np.float32),
        "sample_lin": sam_lin.astype(np.float32),
        "ratio": ratio.astype(np.float32),
        "log_ratio_raw": X_raw.astype(np.float32),
        "feature_after_clip_zscore": X.astype(np.float32),
    }
    return X.astype(np.float32), extras
