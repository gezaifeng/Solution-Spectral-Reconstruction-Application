# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import numpy as np
from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_supported_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTS


def find_images(paths: Iterable[str | Path]) -> List[str]:
    out: list[str] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and is_supported_image(p):
            out.append(str(p))
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and is_supported_image(f):
                    out.append(str(f))
    return sorted(dict.fromkeys(out))


def load_rgb_image(path: str | Path) -> np.ndarray:
    """读取普通图片为 RGB uint8 ndarray: (H,W,3)。V1.0 不支持 RAW/CR2。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not is_supported_image(path):
        raise ValueError(f"不支持的图像格式：{path.suffix}。当前仅支持 JPG/PNG/BMP/TIFF。")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)
