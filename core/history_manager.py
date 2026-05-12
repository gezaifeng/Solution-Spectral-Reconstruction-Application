# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json


def scan_history(output_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(output_root)
    if not root.exists():
        return []
    items = []
    for rec_file in root.rglob("task_record.json"):
        try:
            data = json.loads(rec_file.read_text(encoding="utf-8"))
            task_dir = rec_file.parent
            image_path = data.get("image_path", "")
            copied = data.get("outputs", {}).get("input_image_copy")
            items.append({
                "task_dir": str(task_dir),
                "record_path": str(rec_file),
                "image_path": image_path,
                "image_name": Path(image_path).name if image_path else task_dir.name,
                "created_time": data.get("created_time", ""),
                "num_points": data.get("num_points", ""),
                "feature_mode": data.get("feature_mode", ""),
                "spectrum_csv": str(task_dir / data.get("outputs", {}).get("spectrum_csv", "spectrum.csv")),
                "spectrum_xlsx": str(task_dir / data.get("outputs", {}).get("spectrum_xlsx", "spectrum.xlsx")),
                "spectrum_npy": str(task_dir / data.get("outputs", {}).get("spectrum_npy", "spectrum.npy")),
                "spectrum_png": str(task_dir / data.get("outputs", {}).get("spectrum_png", "spectrum.png")),
                "input_image_copy": str(task_dir / copied) if copied else "",
                "raw": data,
            })
        except Exception:
            continue
    items.sort(key=lambda x: x.get("created_time", ""), reverse=True)
    return items
