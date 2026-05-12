# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, shutil
import numpy as np
import torch
from models.model import ResCNN


class SpectrumPredictor:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.config_path = self.root_dir / "configs" / "model_config.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cfg = None
        self.weight_path = None
        self.output_dim = None
        self.wavelengths = None

    def _read_cfg(self, config_path: str | Path | None = None) -> dict:
        if config_path is None:
            config_path = self.config_path
        config_path = Path(config_path)
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _resolve_weight_path(self, weight_path: str | Path | None = None) -> Path:
        if self.cfg is None:
            self.cfg = self._read_cfg()
        if weight_path is None:
            weight_path = self.cfg.get("weight_file", "weights/best.pth")
        weight_path = Path(weight_path)
        if not weight_path.is_absolute():
            weight_path = self.root_dir / weight_path
        return weight_path

    def inspect_weight(self, weight_path: str | Path) -> dict:
        """检查权重文件，返回输出维度等信息。"""
        weight_path = Path(weight_path)
        if not weight_path.is_absolute():
            weight_path = self.root_dir / weight_path
        if not weight_path.exists():
            raise FileNotFoundError(f"模型权重不存在：{weight_path}")
        ckpt = torch.load(str(weight_path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if not isinstance(state, dict):
            raise ValueError("权重文件格式不正确：未找到 state_dict。")
        out_dim = None
        # ResCNN 默认最后一层为 mlp.3.weight: [out_dim, hidden]
        for key in ("mlp.3.weight", "module.mlp.3.weight"):
            if key in state:
                out_dim = int(state[key].shape[0])
                break
        if out_dim is None:
            # 兜底寻找二维输出层权重
            candidates = [(k, v) for k, v in state.items() if hasattr(v, "ndim") and v.ndim == 2 and ("mlp" in k or "head" in k)]
            if candidates:
                out_dim = int(candidates[-1][1].shape[0])
        if out_dim is None:
            raise ValueError("无法从权重中自动判断输出维度，请确认是否为 ResCNN 权重。")
        return {"output_dim": out_dim, "path": str(weight_path)}

    def update_runtime_config(self, model_name: str, weight_path: str | Path, output_dim: int):
        """更新 configs/model_config.json，并将外部权重复制到 weights/ 目录便于工程自包含。"""
        cfg = self._read_cfg()
        src = Path(weight_path)
        if not src.is_absolute():
            src = self.root_dir / src
        if not src.exists():
            raise FileNotFoundError(f"模型权重不存在：{src}")
        dst_dir = self.root_dir / "weights"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        cfg["model_class"] = model_name
        cfg["weight_file"] = str(Path("weights") / dst.name).replace("\\", "/")
        cfg["output_dim"] = int(output_dim)
        cfg["wavelength_start"] = 380
        cfg["wavelength_stop"] = 380 + 2 * int(output_dim)
        cfg["wavelength_step"] = 2
        self.config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.cfg = cfg

    def load(self, config_path: str | Path | None = None, weight_path: str | Path | None = None):
        if config_path is None:
            config_path = self.config_path
        self.cfg = self._read_cfg(config_path)
        self.weight_path = self._resolve_weight_path(weight_path)
        if not self.weight_path.exists():
            raise FileNotFoundError(f"模型权重不存在：{self.weight_path}")

        # 若权重实际输出维度与配置不一致，以权重为准并兜底更新运行时 output_dim
        info = self.inspect_weight(self.weight_path)
        out_dim = int(info.get("output_dim", self.cfg.get("output_dim", 190)))
        self.output_dim = out_dim
        self.model = ResCNN(
            out_dim=out_dim,
            base_ch=int(self.cfg.get("base_ch", 64)),
            n_blocks=int(self.cfg.get("n_blocks", 6)),
            mlp_hidden=int(self.cfg.get("mlp_hidden", 256)),
            dropout=float(self.cfg.get("dropout", 0.10)),
            out_activation=str(self.cfg.get("out_activation", "linear")),
        ).to(self.device)
        ckpt = torch.load(str(self.weight_path), map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        start = int(self.cfg.get("wavelength_start", 380))
        step = int(self.cfg.get("wavelength_step", 2))
        stop = start + step * out_dim
        self.wavelengths = np.arange(start, stop, step, dtype=np.float32)
        return self

    @torch.no_grad()
    def predict(self, feature_chw: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        x = np.asarray(feature_chw, dtype=np.float32)
        if x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(f"模型输入应为 (3,H,W)，实际为 {x.shape}")
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device).float()
        y = self.model(xt).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return y
