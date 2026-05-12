# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

datas = [
    ("resources", "resources"),
    ("weights", "weights"),
    ("configs", "configs"),
]

hiddenimports = [
    # UI
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",

    # image / data
    "cv2",
    "PIL",
    "PIL.Image",
    "numpy",
    "pandas",
    "openpyxl",

    # plot
    "matplotlib",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.figure",

    # torch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "torch.distributed",
]

excludes = [
    # 可以排除的 PyTorch 生态包
    "torchvision",
    "torchaudio",
    "torchtext",
    "torch.utils.tensorboard",
    "tensorboard",
    "triton",

    # 不需要的额外科学计算/机器学习库
    "sklearn",
    "scikit-learn",
    "scipy",
    "onnx",
    "onnxruntime",
    "pyarrow",

    # 不需要的开发环境
    "IPython",
    "jupyter",
    "notebook",
    "pytest",

    # 其他
    "tkinter",
    "tornado",
    "psutil",
    "gmpy2",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="多色块编码溶液光谱智能重建软件 V1.0",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon="resources/app_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="多色块编码溶液光谱智能重建软件 V1.0",
)