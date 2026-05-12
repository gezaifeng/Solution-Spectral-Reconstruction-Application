# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, traceback, shutil
import numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QTimer, QRect
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QFileDialog, QMessageBox, QStackedWidget, QFrame, QSplitter,
    QSpinBox, QDoubleSpinBox, QTextEdit, QFormLayout, QLineEdit,
    QApplication, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QAbstractSpinBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.image_canvas import ImageCanvas
from core.image_loader import load_rgb_image, find_images
from algorithms.patch_extractor import PatchExtractConfig, extract_rgb_grid
from algorithms.auto_detector import AutoDetectConfig, auto_detect_rects
from algorithms.feature_builder import build_log_ratio_feature
from core.spectrum_predictor import SpectrumPredictor
from core.export_manager import make_task_dir, save_prediction_outputs
from core.history_manager import scan_history

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ImageRecord:
    path: str
    ref_rect: tuple | None = None
    sample_rect: tuple | None = None
    result_dir: str | None = None


class Card(QFrame):
    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(18, 16, 18, 16)
        self.v.setSpacing(10)
        if title:
            lab = QLabel(title)
            lab.setObjectName("CardTitle")
            self.v.addWidget(lab)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setWordWrap(True)
            sub.setObjectName("CardSubTitle")
            self.v.addWidget(sub)



class ElasticButton(QPushButton):
    """首页主按钮/功能按钮的轻量弹性悬停动画。"""
    def __init__(self, text: str = "", parent=None, base_height: int = 42, bump: int = 6):
        super().__init__(text, parent)
        self._base_height = base_height
        self._bump = bump
        self.setMinimumHeight(base_height)
        self._anim = QPropertyAnimation(self, b"minimumHeight", self)
        self._anim.setDuration(230)
        self._anim.setEasingCurve(QEasingCurve.OutBack)

    def _animate_to(self, h: int):
        self._anim.stop()
        self._anim.setStartValue(self.minimumHeight())
        self._anim.setEndValue(h)
        self._anim.start()

    def enterEvent(self, event):
        self._animate_to(self._base_height + self._bump)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_to(self._base_height)
        super().leaveEvent(event)


class NoWheelSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(40)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.setMinimumHeight(40)
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

    def wheelEvent(self, event):
        event.ignore()


class FeatureCard(QFrame):
    """首页流程卡片：只展示流程节点，不再放置重复跳转按钮。"""
    def __init__(self, icon: str, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("FeatureCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 28, 22, 28)
        lay.setSpacing(14)
        lay.setAlignment(Qt.AlignCenter)
        ic = QLabel(icon)
        ic.setObjectName("FeatureIcon")
        ic.setAlignment(Qt.AlignCenter)
        title_lab = QLabel(title)
        title_lab.setObjectName("FeatureTitle")
        title_lab.setAlignment(Qt.AlignCenter)
        lay.addStretch(1)
        lay.addWidget(ic)
        lay.addWidget(title_lab)
        lay.addStretch(1)



class SpectrumPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 3.8), dpi=110)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setMinimumHeight(260)
        self._draw_empty()

    def _draw_empty(self):
        self.ax.clear()
        self.ax.set_title("重构光谱", fontsize=13, weight="bold")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Absorbance")
        self.ax.grid(True, alpha=0.28)
        self.ax.text(0.5, 0.5, "完成预测后将在这里显示重构光谱", ha="center", va="center", transform=self.ax.transAxes, color="#7b7f87")
        self.fig.tight_layout()
        self.draw()

    def plot(self, wavelengths, spectrum, title: str | None = None):
        self.ax.clear()
        self.ax.plot(wavelengths, spectrum, linewidth=2.2)
        self.ax.set_title(title or "重构光谱", fontsize=13, weight="bold")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Absorbance")
        self.ax.grid(True, alpha=0.28)
        if len(spectrum):
            self.ax.text(
                0.02, 0.95,
                f"Range: {wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm\nPoints: {len(wavelengths)}\nMax: {np.max(spectrum):.4g}",
                transform=self.ax.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="#d9dde6")
            )
        self.fig.tight_layout()
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多色块编码溶液光谱智能重建软件 V1.0")
        icon_path = ROOT / "resources" / "app_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1500, 930)
        self.records: list[ImageRecord] = []
        self.current_image: np.ndarray | None = None
        self.current_result: dict | None = None
        self.history_items: list[dict] = []
        self.filtered_history_items: list[dict] = []
        self.home_intro_played = False
        self.home_feature_cards = []
        self._home_anims = []
        self.output_root = ROOT / "outputs" / "tasks"
        self.predictor = SpectrumPredictor(ROOT).load()
        self._build_ui()
        self._apply_style()
        self.log("软件启动完成。当前模型：ResCNN，log_ratio 输入，输出 190 维，波长轴 380–758 nm，2 nm 间隔。")

    def _build_ui(self):
        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(96)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(16, 22, 16, 22)
        side.setSpacing(16)
        logo = QLabel()
        logo.setObjectName("Logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setToolTip("Solution Spectral Reconstruction Application")
        app_icon_png = ROOT / "resources" / "app_icon.png"
        if app_icon_png.exists():
            logo.setPixmap(QPixmap(str(app_icon_png)).scaled(58, 58, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo.setText("SSR")
        side.addWidget(logo)
        side.addSpacing(22)
        self.btn_home = self._side_button("首页")
        self.btn_work = self._side_button("重建")
        self.btn_history = self._side_button("历史")
        self.btn_settings = self._side_button("设置")
        side.addWidget(self.btn_home)
        side.addWidget(self.btn_work)
        side.addWidget(self.btn_history)
        side.addWidget(self.btn_settings)
        side.addStretch(1)
        self.btn_open_out = self._side_button("输出")
        side.addWidget(self.btn_open_out)

        self.stack = QStackedWidget()
        self.home_page = self._build_home_page()
        self.work_page = self._build_work_page()
        self.history_page = self._build_history_page()
        self.settings_page = self._build_settings_page()
        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.work_page)
        self.stack.addWidget(self.history_page)
        self.stack.addWidget(self.settings_page)

        root_layout.addWidget(self.sidebar)
        root_layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        QTimer.singleShot(420, self._play_home_intro_once)

        self.btn_home.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.btn_work.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.btn_history.clicked.connect(lambda: (self.refresh_history(), self.stack.setCurrentIndex(2)))
        self.btn_settings.clicked.connect(lambda: self.stack.setCurrentIndex(3))
        self.btn_open_out.clicked.connect(self.open_output_dir)


    def _play_home_intro_once(self):
        """首页四个流程模块首次打开时的轻量弹性出现动画。"""
        if self.home_intro_played or not getattr(self, "home_feature_cards", None):
            return
        self.home_intro_played = True
        self._home_anims = []
        for i, card in enumerate(self.home_feature_cards):
            end = card.geometry()
            if end.width() <= 0 or end.height() <= 0:
                continue
            start = QRect(
                end.center().x() - int(end.width() * 0.46),
                end.center().y() - int(end.height() * 0.46),
                int(end.width() * 0.92),
                int(end.height() * 0.92),
            )
            anim = QPropertyAnimation(card, b"geometry", self)
            anim.setDuration(520)
            anim.setStartValue(start)
            anim.setEndValue(end)
            anim.setEasingCurve(QEasingCurve.OutBack)
            QTimer.singleShot(90 * i, anim.start)
            self._home_anims.append(anim)

    def _side_button(self, text):
        b = QPushButton(text)
        b.setObjectName("SideButton")
        b.setCursor(Qt.PointingHandCursor)
        return b

    def _show_info(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def _build_home_page(self):
        page = QWidget()
        page.setObjectName("HomePage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(18)

        hero = Card()
        hero.setObjectName("HeroCard")
        h = QVBoxLayout()
        h.setSpacing(18)
        h.setAlignment(Qt.AlignCenter)
        title = QLabel("多色块编码溶液光谱智能重建软件 V1.0")
        title.setObjectName("HeroTitle")
        title.setAlignment(Qt.AlignCenter)
        sub = QLabel("Solution Spectral Reconstruction Application")
        sub.setObjectName("SubTitle")
        sub.setAlignment(Qt.AlignCenter)
        h.addStretch(1)
        h.addWidget(title)
        h.addWidget(sub)
        h.addStretch(1)
        hero.v.addLayout(h)
        layout.addWidget(hero)

        row = QHBoxLayout()
        items = [
            ("📷", "图像输入"),
            ("🎯", "自动/手动标定"),
            ("🧬", "特征构建"),
            ("📈", "光谱重建"),
        ]
        self.home_feature_cards = []
        for icon, t in items:
            fc = FeatureCard(icon, t)
            self.home_feature_cards.append(fc)
            row.addWidget(fc)
        layout.addLayout(row)

        action_card = Card()
        action_col = QVBoxLayout()
        action_col.setContentsMargins(0, 0, 0, 0)
        action_col.setSpacing(12)
        start = ElasticButton("开始光谱重建  →", base_height=46, bump=7)
        start.setObjectName("HeroButton")
        start.setCursor(Qt.PointingHandCursor)
        start.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        hist = ElasticButton("查看历史识别", base_height=42, bump=6)
        hist.setObjectName("NormalButton")
        hist.setCursor(Qt.PointingHandCursor)
        hist.clicked.connect(lambda: (self.refresh_history(), self.stack.setCurrentIndex(2)))
        action_col.addWidget(start)
        action_col.addWidget(hist)
        action_card.v.addLayout(action_col)
        layout.addWidget(action_card)
        layout.addStretch(1)
        return page

    def _home_feature_clicked(self, title: str):
        if title in ("图像输入", "自动/手动标定", "特征构建", "光谱重建"):
            self.stack.setCurrentIndex(1)

    def _home_param_clicked(self, name: str):
        if name == "色卡规格":
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(3)

    def _build_work_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)
        header = QLabel("光谱重建工作台")
        header.setObjectName("PageTitle")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # Left workspace: image + large spectrum below
        left_wrap = QWidget()
        left_l = QVBoxLayout(left_wrap)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.setSpacing(12)
        image_card = Card("图像与目标区域", "在图像中依次框选上方 Ref 和下方 Sample；网格会按当前行列数显示。")
        image_card.v.setContentsMargins(12, 12, 12, 12)
        self.canvas = ImageCanvas()
        self.canvas.rectChanged.connect(self.on_rect_changed)
        image_card.v.addWidget(self.canvas, 1)
        left_l.addWidget(image_card, 3)

        spectrum_card = Card("重建光谱可视化", "预测完成后在此显示完整光谱曲线；PNG/CSV/XLSX/NPY 会同步保存。")
        self.plot = SpectrumPlot()
        spectrum_card.v.addWidget(self.plot)
        srow = QHBoxLayout()
        self.btn_open_last = QPushButton("打开当前结果目录")
        self.btn_open_last.setObjectName("NormalButton")
        self.btn_open_last.clicked.connect(self.open_current_result_dir)
        srow.addStretch(1)
        srow.addWidget(self.btn_open_last)
        spectrum_card.v.addLayout(srow)
        left_l.addWidget(spectrum_card, 2)
        splitter.addWidget(left_wrap)

        # Right control panel in scroll area
        right_content = QWidget()
        right_l = QVBoxLayout(right_content)
        right_l.setContentsMargins(0, 0, 0, 0)
        right_l.setSpacing(12)

        import_card = Card("1. 图像导入")
        row = QHBoxLayout()
        b1 = QPushButton("导入单张")
        b2 = QPushButton("批量导入")
        b1.setObjectName("PrimaryButton")
        b2.setObjectName("NormalButton")
        b1.clicked.connect(self.import_single)
        b2.clicked.connect(self.import_batch)
        row.addWidget(b1); row.addWidget(b2)
        import_card.v.addLayout(row)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(105)
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        import_card.v.addWidget(self.image_list)
        right_l.addWidget(import_card)

        roi_card = Card("2. 区域识别与手动校正")
        auto_row = QHBoxLayout()
        ba = QPushButton("自动识别当前图像")
        ba.setObjectName("PrimaryButton")
        ba.clicked.connect(self.auto_detect_current)
        auto_row.addWidget(ba)
        roi_card.v.addLayout(auto_row)
        row2 = QHBoxLayout()
        self.btn_manual_ref = QPushButton("手动框选 Ref")
        self.btn_manual_sample = QPushButton("手动框选 Sample")
        bc = QPushButton("清除")
        self.btn_manual_ref.setObjectName("BlueButton")
        self.btn_manual_sample.setObjectName("GreenButton")
        bc.setObjectName("NormalButton")
        self.btn_manual_ref.setCheckable(True)
        self.btn_manual_sample.setCheckable(True)
        self.btn_manual_ref.clicked.connect(lambda: self.set_manual_mode("ref"))
        self.btn_manual_sample.clicked.connect(lambda: self.set_manual_mode("sample"))
        bc.clicked.connect(self.clear_current_rects)
        row2.addWidget(self.btn_manual_ref); row2.addWidget(self.btn_manual_sample); row2.addWidget(bc)
        roi_card.v.addLayout(row2)
        self.roi_hint = QLabel("提示：优先点击自动识别；若蓝框/绿框不准确，可继续使用手动框选 Ref 与 Sample 覆盖结果。淡色网格表示实际提取区域。")
        self.roi_hint.setObjectName("Hint")
        self.roi_hint.setWordWrap(True)
        roi_card.v.addWidget(self.roi_hint)
        right_l.addWidget(roi_card)

        param_card = Card("3. 提取参数")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.spin_rows = NoWheelSpinBox(); self.spin_rows.setRange(1, 50); self.spin_rows.setValue(6)
        self.spin_cols = NoWheelSpinBox(); self.spin_cols.setRange(1, 50); self.spin_cols.setValue(12)
        self.spin_sample_count = NoWheelSpinBox(); self.spin_sample_count.setRange(1, 5000); self.spin_sample_count.setValue(100)
        self.spin_center_area = NoWheelDoubleSpinBox(); self.spin_center_area.setRange(0.05, 1.0); self.spin_center_area.setSingleStep(0.05); self.spin_center_area.setDecimals(2); self.spin_center_area.setValue(0.40)
        self.spin_crop_long = NoWheelDoubleSpinBox(); self.spin_crop_long.setRange(0.0, 0.30); self.spin_crop_long.setSingleStep(0.01); self.spin_crop_long.setDecimals(3); self.spin_crop_long.setValue(0.010)
        self.spin_crop_short = NoWheelDoubleSpinBox(); self.spin_crop_short.setRange(0.0, 0.30); self.spin_crop_short.setSingleStep(0.01); self.spin_crop_short.setDecimals(3); self.spin_crop_short.setValue(0.020)
        for sp in [self.spin_rows, self.spin_cols]:
            sp.valueChanged.connect(lambda _: self.canvas.set_grid(self.spin_rows.value(), self.spin_cols.value()))
        for sp in [self.spin_crop_long, self.spin_crop_short]:
            sp.valueChanged.connect(lambda _: self.canvas.set_crop(self.spin_crop_long.value(), self.spin_crop_short.value()))
        form.addRow("色卡行数", self.spin_rows)
        form.addRow("色卡列数", self.spin_cols)
        form.addRow("每块采样点数", self.spin_sample_count)
        form.addRow("中心采样面积", self.spin_center_area)
        form.addRow("黑边内缩-横向", self.spin_crop_long)
        form.addRow("黑边内缩-纵向", self.spin_crop_short)
        param_card.v.addLayout(form)
        hint = QLabel("黑边内缩用于去除目标区域外框/黑边。横向按宽度比例内缩，纵向按高度比例内缩。")
        hint.setWordWrap(True)
        hint.setObjectName("Hint")
        param_card.v.addWidget(hint)
        right_l.addWidget(param_card)

        pred_card = Card("4. 光谱重建")
        bp = QPushButton("预测当前图像")
        bb = QPushButton("批量预测 / 复用当前框")
        ba_batch = QPushButton("批量自动识别并预测")
        bp.setObjectName("PrimaryButtonBig")
        bb.setObjectName("NormalButton")
        ba_batch.setObjectName("NormalButton")
        bp.clicked.connect(self.predict_current)
        bb.clicked.connect(self.predict_batch_reuse_current_roi)
        ba_batch.clicked.connect(self.predict_batch_auto_detect)
        pred_card.v.addWidget(bp)
        pred_card.v.addWidget(bb)
        pred_card.v.addWidget(ba_batch)
        self.progress = QProgressBar()
        self.progress.setObjectName("ModernProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        pred_card.v.addWidget(QLabel("处理进度"))
        pred_card.v.addWidget(self.progress)
        self.output_edit = QLineEdit(str(self.output_root))
        choose_out = QPushButton("选择输出目录")
        choose_out.setObjectName("NormalButton")
        choose_out.clicked.connect(self.choose_output_dir)
        pred_card.v.addWidget(QLabel("输出目录"))
        pred_card.v.addWidget(self.output_edit)
        pred_card.v.addWidget(choose_out)
        right_l.addWidget(pred_card)

        log_card = Card("运行日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        log_card.v.addWidget(self.log_text)
        right_l.addWidget(log_card)
        right_l.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(right_content)
        splitter.addWidget(scroll)
        scroll.setMinimumWidth(500)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([900, 560])
        layout.addWidget(splitter, 1)
        return page


    def _build_history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)
        title = QLabel("历史识别与结果管理")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        top_card = Card("历史记录", "自动保存每次识别与预测的输入图片、中间特征、预测光谱和可视化结果。")
        search_row = QHBoxLayout()
        self.history_search_edit = QLineEdit()
        self.history_search_edit.setPlaceholderText("按时间或文件名查找历史记录...")
        self.btn_search_history = QPushButton("查找")
        self.btn_clear_history_search = QPushButton("清空")
        self.btn_search_history.setObjectName("PrimaryButton")
        self.btn_clear_history_search.setObjectName("NormalButton")
        self.history_search_edit.returnPressed.connect(self.apply_history_filter)
        self.btn_search_history.clicked.connect(self.apply_history_filter)
        self.btn_clear_history_search.clicked.connect(self.clear_history_filter)
        search_row.addWidget(self.history_search_edit, 1)
        search_row.addWidget(self.btn_search_history)
        search_row.addWidget(self.btn_clear_history_search)
        top_card.v.addLayout(search_row)

        top_row = QHBoxLayout()
        self.btn_refresh_history = QPushButton("刷新历史")
        self.btn_import_history = QPushButton("导入历史")
        self.btn_delete_history = QPushButton("删除记录")
        self.btn_open_hist_dir = QPushButton("打开结果目录")
        self.btn_open_hist_png = QPushButton("打开光谱图")
        self.btn_load_hist_image = QPushButton("载入原图到工作台")
        for b in [self.btn_refresh_history, self.btn_import_history, self.btn_delete_history, self.btn_open_hist_dir, self.btn_open_hist_png, self.btn_load_hist_image]:
            b.setObjectName("NormalButton")
            top_row.addWidget(b)
        self.btn_refresh_history.setObjectName("PrimaryButton")
        self.btn_delete_history.setObjectName("DangerButton")
        self.btn_refresh_history.clicked.connect(self.refresh_history)
        self.btn_import_history.clicked.connect(self.import_history_file)
        self.btn_delete_history.clicked.connect(self.delete_selected_history)
        self.btn_open_hist_dir.clicked.connect(self.open_selected_history_dir)
        self.btn_open_hist_png.clicked.connect(self.open_selected_history_png)
        self.btn_load_hist_image.clicked.connect(self.load_selected_history_image)
        top_card.v.addLayout(top_row)
        layout.addWidget(top_card)

        splitter = QSplitter(Qt.Horizontal)
        left = Card("记录列表")
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["时间", "图像", "点数", "目录"])
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.currentCellChanged.connect(lambda *_: self.on_history_selected())
        left.v.addWidget(self.history_table)
        splitter.addWidget(left)

        right = Card("历史复查预览", "选择左侧记录后，同时查看原图识别区域和重构光谱。")
        self.history_detail = QLabel("暂无历史记录。")
        self.history_detail.setObjectName("BodyText")
        self.history_detail.setWordWrap(True)
        self.history_canvas = ImageCanvas()
        self.history_canvas.setMinimumHeight(300)
        self.history_plot = SpectrumPlot()
        right.v.addWidget(self.history_detail)
        right.v.addWidget(self.history_canvas, 1)
        right.v.addWidget(self.history_plot, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)
        return page

    def _build_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(14)
        title = QLabel("系统设置")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        c = Card("当前模型配置", "可在此更新模型名称、替换权重文件，并自动检查权重输出维度。")
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        self.model_name_edit = QLineEdit(str(self.predictor.cfg.get("model_class", "ResCNN")))
        self.weight_path_edit = QLineEdit(str(self.predictor.weight_path))
        self.weight_path_edit.setReadOnly(True)
        self.out_dim_edit = QLineEdit(str(self.predictor.output_dim))
        self.out_dim_edit.setReadOnly(True)
        self.wave_axis_edit = QLineEdit(f"np.arange({int(self.predictor.wavelengths[0])}, {int(self.predictor.wavelengths[-1])+2}, 2)")
        self.wave_axis_edit.setReadOnly(True)
        self.preprocess_edit = QLineEdit("log_ratio + clip[-8,8] + per-image channel z-score")
        self.preprocess_edit.setReadOnly(True)
        form.addRow("模型名称", self.model_name_edit)
        form.addRow("权重文件", self.weight_path_edit)
        form.addRow("输出维度", self.out_dim_edit)
        form.addRow("波长轴", self.wave_axis_edit)
        form.addRow("预处理", self.preprocess_edit)
        c.v.addLayout(form)
        btn_row = QHBoxLayout()
        self.btn_choose_weight = QPushButton("上传/选择新权重")
        self.btn_check_weight = QPushButton("检查权重维度")
        self.btn_apply_model = QPushButton("应用模型设置")
        self.btn_choose_weight.setObjectName("NormalButton")
        self.btn_check_weight.setObjectName("NormalButton")
        self.btn_apply_model.setObjectName("PrimaryButton")
        self.btn_choose_weight.clicked.connect(self.choose_new_weight)
        self.btn_check_weight.clicked.connect(self.check_current_weight)
        self.btn_apply_model.clicked.connect(self.apply_model_settings)
        btn_row.addWidget(self.btn_choose_weight)
        btn_row.addWidget(self.btn_check_weight)
        btn_row.addWidget(self.btn_apply_model)
        c.v.addLayout(btn_row)
        note = QLabel("说明：当前软件只接入 ResCNN 模型结构；更换权重时会检查最后输出层维度，并自动更新输出维度与波长轴长度。若后续要替换为全新模型结构，可在此接口基础上扩展 model.py。")
        note.setObjectName("Hint")
        note.setWordWrap(True)
        c.v.addWidget(note)
        layout.addWidget(c)
        layout.addStretch(1)
        return page


    def choose_new_weight(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "PyTorch Weights (*.pth *.pt);;All Files (*)")
        if not f:
            return
        try:
            info = self.predictor.inspect_weight(f)
            self.weight_path_edit.setText(f)
            self.out_dim_edit.setText(str(info["output_dim"]))
            self.wave_axis_edit.setText(f"np.arange(380, {380 + 2 * int(info['output_dim'])}, 2)")
            self.log(f"新权重检查通过：输出维度 {info['output_dim']}，文件 {f}")
        except Exception as e:
            QMessageBox.critical(self, "权重检查失败", str(e))

    def check_current_weight(self):
        try:
            info = self.predictor.inspect_weight(self.weight_path_edit.text())
            self.out_dim_edit.setText(str(info["output_dim"]))
            self.wave_axis_edit.setText(f"np.arange(380, {380 + 2 * int(info['output_dim'])}, 2)")
            QMessageBox.information(self, "检查完成", f"权重格式有效，检测到输出维度：{info['output_dim']}")
        except Exception as e:
            QMessageBox.critical(self, "权重检查失败", str(e))

    def apply_model_settings(self):
        try:
            model_name = self.model_name_edit.text().strip() or "ResCNN"
            weight_path = self.weight_path_edit.text().strip()
            info = self.predictor.inspect_weight(weight_path)
            out_dim = int(info["output_dim"])
            self.predictor.update_runtime_config(model_name=model_name, weight_path=weight_path, output_dim=out_dim)
            self.predictor.load()
            self.out_dim_edit.setText(str(self.predictor.output_dim))
            self.wave_axis_edit.setText(f"np.arange(380, {380 + 2 * int(self.predictor.output_dim)}, 2)")
            QMessageBox.information(self, "应用成功", f"模型设置已更新。当前输出维度：{self.predictor.output_dim}")
            self.log(f"模型设置已更新：{model_name}, 权重={weight_path}, 输出维度={self.predictor.output_dim}")
        except Exception as e:
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "应用失败", str(e))

    def _apply_style(self):
        self.setStyleSheet('''
            QWidget { background: #f4f5f7; color: #16181d; font-family: "Microsoft YaHei", "Segoe UI"; font-size: 14px; }
            #HomePage { border-image: url(resources/home_background.png) 0 0 0 0 stretch stretch; }
            #Sidebar { background: #07080a; border-radius: 0px; }
            #Logo { background: #ffffff; border-radius: 18px; min-height: 64px; min-width: 64px; padding: 4px; }
            #SideButton { background: transparent; color: #f2f2f2; border: none; border-radius: 16px; min-height: 50px; font-weight: 700; }
            #SideButton:hover { background: #202227; }
            #Card, #FeatureCard { background: rgba(255,255,255,0.88); border: 1px solid #e6e8ef; border-radius: 24px; }
            #HeroCard { background: rgba(255,255,255,0.72); border: 1px solid rgba(226,232,244,0.92); border-radius: 28px; }
            #CardTitle { font-size: 17px; font-weight: 900; color: #1d2028; background: transparent; }
            #CardSubTitle { font-size: 13px; color: #687181; background: transparent; }
            #PageTitle { font-size: 28px; font-weight: 900; background: transparent; }
            #HeroTitle { font-size: 32px; font-weight: 900; background: transparent; }
            #SubTitle { font-size: 17px; color: #6b7280; background: transparent; }
            #HeroDesc { font-size: 16px; color: #374151; background: transparent; line-height: 1.6; }
            #BodyText, #FlowText, #Hint { color: #4b5563; background: transparent; line-height: 1.5; }
            #FeatureIcon { font-size: 38px; background: transparent; }
            #FeatureTitle { font-size: 19px; font-weight: 900; background: transparent; }
            #StatButton { background: rgba(255,255,255,0.72); border: 1px solid #dfe5ef; border-radius: 18px; padding: 14px; font-size: 14px; min-height: 58px; }
            #StatButton:hover { background: #edf2ff; }
            QPushButton { border: none; border-radius: 12px; padding: 10px 14px; font-weight: 800; }
            #HeroButton { background: #111111; color: white; min-height: 44px; font-size: 16px; border-radius: 14px; }
            #PrimaryButton, #PrimaryButtonBig { background: #111111; color: white; }
            #PrimaryButtonBig { min-height: 42px; font-size: 15px; }
            #PrimaryButton:hover, #PrimaryButtonBig:hover, #HeroButton:hover { background: #2b2b2b; }
            #NormalButton, #LightButton { background: #edf0f5; color: #1f2937; }
            #NormalButton:hover, #LightButton:hover { background: #e0e5ee; }
            #BlueButton { background: #e8f1ff; color: #1f66cc; }
            #BlueButton:checked { background: #2f80ed; color: white; border: 2px solid #0f55b5; }
            #GreenButton { background: #e8f8ef; color: #16864d; }
            #GreenButton:checked { background: #27ae60; color: white; border: 2px solid #0f7b3a; }
            #DangerButton { background: #fff0f0; color: #b42318; }
            #DangerButton:hover { background: #ffe1e1; }
            QListWidget, QTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox { background: #fbfcfe; border: 1px solid #dfe3eb; border-radius: 12px; padding: 6px 36px 6px 10px; min-height: 28px; }
            QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 30px; border-left: 1px solid #dfe3eb; border-bottom: 1px solid #edf0f5; border-top-right-radius: 12px; background: #f1f4f9; }
            QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 30px; border-left: 1px solid #dfe3eb; border-bottom-right-radius: 12px; background: #f1f4f9; }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #e0e7f2; }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 7px solid #334155; }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 7px solid #334155; }
            QScrollArea { background: transparent; }
            QProgressBar#ModernProgress { background: #edf0f5; border: none; border-radius: 10px; height: 20px; text-align: center; color: #1f2937; font-weight: 800; }
            QProgressBar#ModernProgress::chunk { background: #111111; border-radius: 10px; }
            QSplitter::handle { background: transparent; width: 10px; height: 10px; }
            QScrollBar:vertical { background: #edf0f5; width: 12px; margin: 4px 2px 4px 2px; border-radius: 6px; }
            QScrollBar::handle:vertical { background: #cfd5df; min-height: 38px; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #aeb8c7; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; background: transparent; }
            QScrollBar:horizontal { background: #edf0f5; height: 12px; margin: 2px 4px 2px 4px; border-radius: 6px; }
            QScrollBar::handle:horizontal { background: #cfd5df; min-width: 38px; border-radius: 6px; }
            QScrollBar::handle:horizontal:hover { background: #aeb8c7; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; background: transparent; }
        ''')

    def log(self, msg: str):
        if hasattr(self, 'log_text'):
            self.log_text.append(msg)
        print(msg)

    def import_single(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择图像", "", "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)")
        if files:
            self.add_images(files)

    def import_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            imgs = find_images([folder])
            self.add_images(imgs)

    def add_images(self, files):
        existing = {r.path for r in self.records}
        added = 0
        for f in files:
            if f not in existing:
                self.records.append(ImageRecord(path=f))
                self.image_list.addItem(Path(f).name)
                added += 1
        if added:
            self.log(f"导入图像 {added} 张。")
            if self.image_list.currentRow() < 0:
                self.image_list.setCurrentRow(0)
        else:
            self.log("没有新增图像。")

    def current_record(self) -> ImageRecord | None:
        row = self.image_list.currentRow()
        if 0 <= row < len(self.records):
            return self.records[row]
        return None

    def on_image_selected(self, row: int):
        rec = self.current_record()
        if rec is None:
            return
        try:
            self.current_image = load_rgb_image(rec.path)
            self.canvas.set_image(self.current_image)
            self.canvas.set_rects(rec.ref_rect, rec.sample_rect)
            self.canvas.set_grid(self.spin_rows.value(), self.spin_cols.value())
            self.canvas.set_crop(self.spin_crop_long.value(), self.spin_crop_short.value())
            self.log(f"当前图像：{Path(rec.path).name}")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", str(e))


    def set_manual_mode(self, mode: str | None):
        """设置手动框选模式，并同步按钮高亮反馈。"""
        self.canvas.set_mode(mode)
        if hasattr(self, "btn_manual_ref"):
            self.btn_manual_ref.setChecked(mode == "ref")
        if hasattr(self, "btn_manual_sample"):
            self.btn_manual_sample.setChecked(mode == "sample")
        if mode == "ref":
            self.roi_hint.setText("当前模式：正在框选上方 Ref 参考区域。请在图像中按住左键拖拽绘制矩形。")
        elif mode == "sample":
            self.roi_hint.setText("当前模式：正在框选下方 Sample 样本区域。请在图像中按住左键拖拽绘制矩形。")
        else:
            self.roi_hint.setText("提示：优先点击自动识别；若蓝框/绿框不准确，可继续使用手动框选 Ref 与 Sample 覆盖结果。淡色网格表示实际提取区域。")

    def on_rect_changed(self, kind: str, rect: tuple):
        rec = self.current_record()
        if rec is None:
            return
        if kind == "ref":
            rec.ref_rect = tuple(rect)
            self.log(f"Ref 区域已设置：{rec.ref_rect}")
            self.set_manual_mode(None)
        elif kind == "sample":
            rec.sample_rect = tuple(rect)
            self.log(f"Sample 区域已设置：{rec.sample_rect}")
            self.set_manual_mode(None)

    def clear_current_rects(self):
        rec = self.current_record()
        if rec:
            rec.ref_rect = None
            rec.sample_rect = None
            self.canvas.clear_rects()
            self.set_manual_mode(None)
            self.log("已清除当前图像的 Ref/Sample 框选。")

    def choose_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if d:
            self.output_edit.setText(d)
            self.output_root = Path(d)

    def open_output_dir(self):
        d = Path(self.output_edit.text()) if hasattr(self, 'output_edit') else self.output_root
        d.mkdir(parents=True, exist_ok=True)
        if os.name == 'nt':
            os.startfile(str(d))  # type: ignore
        else:
            os.system(f'xdg-open "{d}"')

    def open_current_result_dir(self):
        if self.current_result and self.current_result.get("task_dir"):
            d = Path(self.current_result["task_dir"])
        else:
            rec = self.current_record()
            d = Path(rec.result_dir) if rec and rec.result_dir else None
        if d and d.exists():
            if os.name == 'nt':
                os.startfile(str(d))  # type: ignore
            else:
                os.system(f'xdg-open "{d}"')
        else:
            QMessageBox.information(self, "提示", "当前还没有可打开的预测结果目录。")


    def _auto_cfg(self) -> AutoDetectConfig:
        """自动识别参数，默认对接你原 detect.py 的 Sobel + 连通域筛选思路。"""
        return AutoDetectConfig(
            target_height=512,
            sobel_ksize=3,
            edge_thresh=50,
            det_auto_brighten=True,
            bright_target_median=110.0,
            bright_max_gain=4.0,
            min_cc_area=0.005,
            cc_topk=10,
            pair_center_constraint=True,
            pair_center_band=0.30,
            pair_min_dy=0.12,
            pair_max_area_ratio=1.35,
            pair_max_ar_ratio=1.25,
        )

    def auto_detect_current(self):
        rec = self.current_record()
        if rec is None:
            QMessageBox.warning(self, "提示", "请先导入并选择一张图像。")
            return
        try:
            image = load_rgb_image(rec.path)
            ref_rect, sample_rect, _ = auto_detect_rects(image, self._auto_cfg())
            if ref_rect is None or sample_rect is None:
                self.log(f"[自动识别失败] {Path(rec.path).name}，请使用手动框选。")
                return
            rec.ref_rect = tuple(ref_rect)
            rec.sample_rect = tuple(sample_rect)
            self.canvas.set_rects(rec.ref_rect, rec.sample_rect)
            self.canvas.set_grid(self.spin_rows.value(), self.spin_cols.value())
            self.canvas.set_crop(self.spin_crop_long.value(), self.spin_crop_short.value())
            self.log(f"[自动识别成功] {Path(rec.path).name} Ref={rec.ref_rect} Sample={rec.sample_rect}")
        except Exception as e:
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "自动识别异常", str(e))

    def predict_batch_auto_detect(self):
        if not self.records:
            QMessageBox.warning(self, "提示", "请先导入单张或批量图像。")
            return
        ok, fail = 0, 0
        last = None
        last_name = "样本"
        total = max(1, len(self.records))
        self.progress.setValue(0)
        for idx, rec in enumerate(self.records, 1):
            try:
                image = load_rgb_image(rec.path)
                ref_rect, sample_rect, _ = auto_detect_rects(image, self._auto_cfg())
                if ref_rect is None or sample_rect is None:
                    raise ValueError("自动识别失败，未获得 Ref/Sample 两个区域")
                rec.ref_rect = tuple(ref_rect)
                rec.sample_rect = tuple(sample_rect)
                wavelengths, spectrum, paths = self._process_one(rec)
                ok += 1
                last = (wavelengths, spectrum)
                last_name = Path(rec.path).stem
                self.current_result = paths
                self.log(f"[自动批量成功] {Path(rec.path).name} → {paths['task_dir']}")
                self.progress.setValue(int(idx / total * 100))
                QApplication.processEvents()
            except Exception as e:
                fail += 1
                self.log(f"[自动批量失败] {Path(rec.path).name}: {e}")
        if last:
            self.plot.plot(last[0], last[1], title=f"{last_name}-重构光谱")
        self.progress.setValue(100 if ok else 0)
        self.refresh_history()
        QMessageBox.information(self, "自动批量完成", f"成功 {ok} 张，失败 {fail} 张。失败图片仍可切换为手动框选后单独预测。")

    def refresh_history(self):
        self.history_items = scan_history(self.output_edit.text() if hasattr(self, 'output_edit') else self.output_root)
        query = self.history_search_edit.text().strip() if hasattr(self, 'history_search_edit') else ""
        self.filtered_history_items = self._filter_history_items(query)
        if not hasattr(self, 'history_table'):
            return
        self.history_table.setRowCount(len(self.filtered_history_items))
        for row, item in enumerate(self.filtered_history_items):
            vals = [item.get("created_time", ""), item.get("image_name", ""), str(item.get("num_points", "")), item.get("task_dir", "")]
            for col, val in enumerate(vals):
                self.history_table.setItem(row, col, QTableWidgetItem(val))
        if self.filtered_history_items:
            self.history_table.setCurrentCell(0, 0)
            self.on_history_selected()
        else:
            self.history_detail.setText("暂无历史记录。完成一次预测后，这里会自动列出输入图像、中间特征与光谱结果。")
            self.history_canvas.clear_rects()
            self.history_plot._draw_empty()


    def _filter_history_items(self, query: str):
        q = (query or "").strip().lower()
        if not q:
            return list(self.history_items)
        out = []
        for item in self.history_items:
            hay = " ".join([
                str(item.get("created_time", "")),
                str(item.get("image_name", "")),
                str(item.get("task_dir", "")),
            ]).lower()
            if q in hay:
                out.append(item)
        return out

    def apply_history_filter(self):
        if not hasattr(self, "history_table"):
            return
        query = self.history_search_edit.text().strip() if hasattr(self, "history_search_edit") else ""
        self.filtered_history_items = self._filter_history_items(query)
        self.history_table.setRowCount(len(self.filtered_history_items))
        for row, item in enumerate(self.filtered_history_items):
            vals = [item.get("created_time", ""), item.get("image_name", ""), str(item.get("num_points", "")), item.get("task_dir", "")]
            for col, val in enumerate(vals):
                self.history_table.setItem(row, col, QTableWidgetItem(val))
        if self.filtered_history_items:
            self.history_table.setCurrentCell(0, 0)
            self.on_history_selected()
        else:
            self.history_detail.setText("未找到匹配的历史记录。")
            self.history_canvas.clear_rects()
            self.history_plot._draw_empty()

    def clear_history_filter(self):
        if hasattr(self, "history_search_edit"):
            self.history_search_edit.clear()
        self.refresh_history()

    def delete_selected_history(self):
        item = self._selected_history()
        if not item:
            QMessageBox.information(self, "提示", "请先选择一条历史记录。")
            return
        d = Path(item.get("task_dir", ""))
        if not d.exists():
            QMessageBox.information(self, "提示", "该历史目录已不存在，将刷新列表。")
            self.refresh_history()
            return
        ret = QMessageBox.question(self, "确认删除", f"确定删除这条历史记录及其结果文件吗？\n{d}")
        if ret != QMessageBox.Yes:
            return
        try:
            shutil.rmtree(str(d))
            self.log(f"已删除历史记录：{d}")
            self.refresh_history()
        except Exception as e:
            QMessageBox.critical(self, "删除失败", str(e))

    def import_history_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self,
            "导入历史记录",
            "",
            "History Files (*.xlsx *.xls *.csv *.json);;All Files (*)"
        )
        if not f:
            return
        try:
            src_paths = []
            suffix = Path(f).suffix.lower()
            if suffix in (".xlsx", ".xls"):
                import pandas as pd
                df = pd.read_excel(f)
                cols = {str(c).lower(): c for c in df.columns}
                key = None
                for cand in ["task_dir", "目录", "result_dir", "record_path", "历史目录"]:
                    if cand.lower() in cols:
                        key = cols[cand.lower()]
                        break
                if key is None:
                    raise ValueError("Excel 中未找到 task_dir/目录/result_dir/record_path 等历史目录列。")
                src_paths = [str(v) for v in df[key].dropna().tolist()]
            elif suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(f, encoding="utf-8-sig")
                cols = {str(c).lower(): c for c in df.columns}
                key = None
                for cand in ["task_dir", "目录", "result_dir", "record_path", "历史目录"]:
                    if cand.lower() in cols:
                        key = cols[cand.lower()]
                        break
                if key is None:
                    raise ValueError("CSV 中未找到 task_dir/目录/result_dir/record_path 等历史目录列。")
                src_paths = [str(v) for v in df[key].dropna().tolist()]
            elif suffix == ".json":
                p = Path(f)
                if p.name == "task_record.json":
                    src_paths = [str(p.parent)]
                else:
                    import json
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        src_paths = [str(data.get("task_dir") or data.get("record_path") or p.parent)]
                    elif isinstance(data, list):
                        src_paths = [str(x.get("task_dir") or x.get("record_path")) for x in data if isinstance(x, dict)]
            else:
                raise ValueError("暂只支持 xlsx/xls/csv/json 历史索引文件。")

            imported = 0
            out_root = Path(self.output_edit.text()) if hasattr(self, "output_edit") else self.output_root
            out_root.mkdir(parents=True, exist_ok=True)
            for sp in src_paths:
                if not sp:
                    continue
                src = Path(sp)
                if src.is_file() and src.name == "task_record.json":
                    src = src.parent
                if not (src / "task_record.json").exists():
                    continue
                dst = out_root / src.name
                if dst.resolve() == src.resolve():
                    imported += 1
                    continue
                base = dst
                k = 1
                while dst.exists():
                    dst = out_root / f"{base.name}_imported_{k}"
                    k += 1
                shutil.copytree(str(src), str(dst))
                imported += 1
            self.refresh_history()
            QMessageBox.information(self, "导入完成", f"成功导入/关联 {imported} 条历史记录。")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))

    def _selected_history(self):
        if not hasattr(self, 'history_table'):
            return None
        row = self.history_table.currentRow()
        if 0 <= row < len(self.filtered_history_items):
            return self.filtered_history_items[row]
        return None

    def on_history_selected(self):
        item = self._selected_history()
        if not item:
            return
        detail = (
            f"图像：{item.get('image_name','')}\n"
            f"时间：{item.get('created_time','')}\n"
            f"特征：{item.get('feature_mode','log_ratio')}\n"
            f"光谱点数：{item.get('num_points','')}\n"
            f"目录：{item.get('task_dir','')}"
        )
        self.history_detail.setText(detail)
        try:
            image_path = item.get("input_image_copy") or item.get("image_path")
            if image_path and Path(image_path).exists():
                img = load_rgb_image(image_path)
                raw = item.get("raw", {})
                self.history_canvas.set_image(img)
                self.history_canvas.set_rects(tuple(raw.get("rect_ref")) if raw.get("rect_ref") else None,
                                             tuple(raw.get("rect_sample")) if raw.get("rect_sample") else None)
                cfg = raw.get("config", {})
                self.history_canvas.set_grid(int(cfg.get("rows", 6)), int(cfg.get("cols", 12)))
                self.history_canvas.set_crop(float(cfg.get("card_crop_long", 0.01)), float(cfg.get("card_crop_short", 0.02)))
            else:
                self.history_canvas.clear_rects()
        except Exception:
            self.history_canvas.clear_rects()
        try:
            spectrum = np.load(item["spectrum_npy"]).astype(np.float32)
            # 当前模型固定为 np.arange(380,760,2)。若后续换模型，可从 task_record.json 扩展读取。
            wavelengths = np.arange(380, 760, 2, dtype=np.float32)
            if len(wavelengths) != len(spectrum):
                wavelengths = np.arange(len(spectrum), dtype=np.float32)
            self.history_plot.plot(wavelengths, spectrum, title=f"{item.get('image_name', '样本')}-重构光谱")
        except Exception:
            self.history_plot._draw_empty()

    def open_selected_history_dir(self):
        item = self._selected_history()
        if not item:
            QMessageBox.information(self, "提示", "请先选择一条历史记录。")
            return
        d = Path(item["task_dir"])
        if d.exists():
            if os.name == 'nt':
                os.startfile(str(d))  # type: ignore
            else:
                os.system(f'xdg-open "{d}"')

    def open_selected_history_png(self):
        item = self._selected_history()
        if not item:
            QMessageBox.information(self, "提示", "请先选择一条历史记录。")
            return
        p = Path(item["spectrum_png"])
        if p.exists():
            if os.name == 'nt':
                os.startfile(str(p))  # type: ignore
            else:
                os.system(f'xdg-open "{p}"')
        else:
            QMessageBox.information(self, "提示", "该历史记录未找到 spectrum.png。")

    def load_selected_history_image(self):
        item = self._selected_history()
        if not item:
            QMessageBox.information(self, "提示", "请先选择一条历史记录。")
            return
        image_path = item.get("input_image_copy") or item.get("image_path")
        if not image_path or not Path(image_path).exists():
            image_path = item.get("image_path")
        if image_path and Path(image_path).exists():
            self.add_images([image_path])
            self.image_list.setCurrentRow(len(self.records) - 1)
            raw = item.get("raw", {})
            rr = raw.get("rect_ref")
            sr = raw.get("rect_sample")
            rec = self.current_record()
            if rec and rr and sr:
                rec.ref_rect = tuple(rr)
                rec.sample_rect = tuple(sr)
                self.canvas.set_rects(rec.ref_rect, rec.sample_rect)
            self.stack.setCurrentIndex(1)
            self.log(f"已从历史记录载入图像：{Path(image_path).name}")
        else:
            QMessageBox.information(self, "提示", "历史记录中的原始图像不存在。")

    def _patch_cfg(self) -> PatchExtractConfig:
        return PatchExtractConfig(
            rows=self.spin_rows.value(),
            cols=self.spin_cols.value(),
            sample_count=self.spin_sample_count.value(),
            sample_center_area=self.spin_center_area.value(),
            card_crop_long=self.spin_crop_long.value(),
            card_crop_short=self.spin_crop_short.value(),
            random_seed=0,
        )

    def _process_one(self, rec: ImageRecord, ref_rect=None, sample_rect=None, scale_from=None):
        image = load_rgb_image(rec.path)
        rr = ref_rect or rec.ref_rect
        sr = sample_rect or rec.sample_rect
        if rr is None or sr is None:
            raise ValueError("请先完成 Ref 和 Sample 两个区域的手动框选。")
        if scale_from is not None:
            base_w, base_h = scale_from
            h, w = image.shape[:2]
            sx, sy = w / base_w, h / base_h
            rr = tuple([int(round(rr[i] * (sx if i % 2 == 0 else sy))) for i in range(4)])
            sr = tuple([int(round(sr[i] * (sx if i % 2 == 0 else sy))) for i in range(4)])
        cfg = self._patch_cfg()
        ref_rgb = extract_rgb_grid(image, rr, cfg)
        sample_rgb = extract_rgb_grid(image, sr, cfg)
        feature, extras = build_log_ratio_feature(ref_rgb, sample_rgb, clip_value=8.0, use_zscore=True)
        spectrum = self.predictor.predict(feature)
        wavelengths = self.predictor.wavelengths
        task_dir = make_task_dir(self.output_edit.text(), rec.path)
        paths = save_prediction_outputs(
            task_dir,
            image_path=rec.path,
            wavelengths=wavelengths,
            spectrum=spectrum,
            ref_rgb=ref_rgb,
            sample_rgb=sample_rgb,
            feature=feature,
            extras=extras,
            rect_ref=rr,
            rect_sample=sr,
            config={
                "rows": cfg.rows,
                "cols": cfg.cols,
                "sample_count": cfg.sample_count,
                "sample_center_area": cfg.sample_center_area,
                "card_crop_long": cfg.card_crop_long,
                "card_crop_short": cfg.card_crop_short,
            },
        )
        rec.result_dir = paths["task_dir"]
        return wavelengths, spectrum, paths

    def predict_current(self):
        rec = self.current_record()
        if rec is None:
            QMessageBox.warning(self, "提示", "请先导入并选择一张图像。")
            return
        try:
            self.progress.setValue(10)
            wavelengths, spectrum, paths = self._process_one(rec)
            self.plot.plot(wavelengths, spectrum, title=f"{Path(rec.path).stem}-重构光谱")
            self.current_result = paths
            self.log(f"预测完成：{Path(rec.path).name}")
            self.log(f"结果目录：{paths['task_dir']}")
            self.progress.setValue(100)
            self.refresh_history()
        except Exception as e:
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "预测失败", str(e))

    def predict_batch_reuse_current_roi(self):
        cur = self.current_record()
        if cur is None or cur.ref_rect is None or cur.sample_rect is None:
            QMessageBox.warning(self, "提示", "请先在当前图像上框选 Ref 和 Sample，再批量复用当前框。")
            return
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "当前图像未加载。")
            return
        base_h, base_w = self.current_image.shape[:2]
        ok, fail = 0, 0
        last = None
        last_name = "样本"
        total = max(1, len(self.records))
        self.progress.setValue(0)
        for idx, rec in enumerate(self.records, 1):
            try:
                wavelengths, spectrum, paths = self._process_one(
                    rec,
                    ref_rect=cur.ref_rect,
                    sample_rect=cur.sample_rect,
                    scale_from=(base_w, base_h),
                )
                ok += 1
                last = (wavelengths, spectrum)
                last_name = Path(rec.path).stem
                self.current_result = paths
                self.log(f"[批量成功] {Path(rec.path).name} → {paths['task_dir']}")
                self.progress.setValue(int(idx / total * 100))
                QApplication.processEvents()
            except Exception as e:
                fail += 1
                self.log(f"[批量失败] {Path(rec.path).name}: {e}")
        if last:
            self.plot.plot(last[0], last[1], title=f"{last_name}-重构光谱")
        self.progress.setValue(100 if ok else 0)
        self.refresh_history()
        QMessageBox.information(self, "批量完成", f"成功 {ok} 张，失败 {fail} 张。")
