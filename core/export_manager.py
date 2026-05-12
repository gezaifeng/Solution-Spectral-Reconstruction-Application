# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, time, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt


def make_task_dir(output_root: str | Path, image_path: str | Path) -> Path:
    output_root = Path(output_root)
    image_path = Path(image_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = image_path.stem.replace(" ", "_")
    task_dir = output_root / f"{stamp}_{safe_name}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def save_prediction_outputs(task_dir: str | Path, *, image_path: str, wavelengths: np.ndarray,
                            spectrum: np.ndarray, ref_rgb: np.ndarray, sample_rgb: np.ndarray,
                            feature: np.ndarray, extras: dict, rect_ref, rect_sample,
                            config: dict | None = None) -> dict:
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    # 保存原始输入图像副本，保证历史记录不依赖外部路径。
    input_copy_name = None
    try:
        src = Path(image_path)
        if src.exists():
            input_copy_name = "input_image" + src.suffix.lower()
            shutil.copy2(src, task_dir / input_copy_name)
    except Exception:
        input_copy_name = None

    np.save(task_dir / "ref_rgb.npy", ref_rgb.astype(np.float32))
    np.save(task_dir / "sample_rgb.npy", sample_rgb.astype(np.float32))
    np.save(task_dir / "features_log_ratio.npy", feature.astype(np.float32))
    np.save(task_dir / "spectrum.npy", spectrum.astype(np.float32))
    for k, v in extras.items():
        if isinstance(v, np.ndarray):
            np.save(task_dir / f"extra_{k}.npy", v.astype(np.float32))

    df = pd.DataFrame({"Wavelength_nm": wavelengths.astype(np.float32), "Absorbance": spectrum.astype(np.float32)})
    csv_path = task_dir / "spectrum.csv"
    xlsx_path = task_dir / "spectrum.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False)

    fig_path = task_dir / "spectrum.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(wavelengths, spectrum, linewidth=2)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    sample_title = f"{Path(image_path).stem}-重构光谱"
    plt.title(sample_title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    record = {
        "image_path": str(image_path),
        "rect_ref": list(map(int, rect_ref)) if rect_ref is not None else None,
        "rect_sample": list(map(int, rect_sample)) if rect_sample is not None else None,
        "wavelength_start": float(wavelengths[0]) if len(wavelengths) else None,
        "wavelength_end": float(wavelengths[-1]) if len(wavelengths) else None,
        "num_points": int(len(wavelengths)),
        "feature_mode": "log_ratio",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "outputs": {
            "ref_rgb": "ref_rgb.npy",
            "sample_rgb": "sample_rgb.npy",
            "feature": "features_log_ratio.npy",
            "spectrum_npy": "spectrum.npy",
            "spectrum_csv": "spectrum.csv",
            "spectrum_xlsx": "spectrum.xlsx",
            "spectrum_png": "spectrum.png",
            "input_image_copy": input_copy_name,
        }
    }
    (task_dir / "task_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"task_dir": str(task_dir), "csv": str(csv_path), "xlsx": str(xlsx_path), "npy": str(task_dir / "spectrum.npy"), "png": str(fig_path), "input_image_copy": str(task_dir / input_copy_name) if input_copy_name else ""}
