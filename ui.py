"""Main UI components with educational tooltips and info panels."""

import math
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QTabWidget, QScrollArea, QSizePolicy,
    QTextEdit, QSplitter
)
from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QPainter, QPen, QPainterPath

from theme import ThemeColors
from config import (
    ARCHITECTURE_DESCRIPTIONS, DATASET_DESCRIPTIONS,
    TRAINING_TIPS, HYPERPARAMETER_TOOLTIPS, OPTIMIZER_TOOLTIPS,
    ARCHITECTURE_INFO, GAN_MODES
)
from models_pytorch import MLP, CNN, RNN, LSTM, Transformer, Generator, Discriminator
from datasets import MLP_DATASETS, CNN_DATASETS, SEQ_DATASETS, make_gan_data, train_test_split
from factory import make_model, count_parameters
from training import TrainThread
from plots import (
    LossPlot, MetricPlot, LRPlot, GradFlowPlot,
    OutputPlot, ConfusionMatrixPlot, FeatureSpacePlot, WeightDistributionPlot
)

T = ThemeColors


class ArchWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.network_type = "mlp"
        self.layers = None
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_type(self, network_type: str, layers=None):
        self.network_type = network_type
        self.layers = layers
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), T.qc(T.S1))
        
        draw_methods = {
            "mlp": self._draw_mlp, "cnn": self._draw_cnn,
            "rnn": self._draw_rnn, "lstm": self._draw_lstm,
            "transformer": self._draw_transformer, "gan": self._draw_gan,
        }
        draw_methods.get(self.network_type, self._draw_mlp)(p, self.width(), self.height())
        self._draw_badge(p, self.width() - 72, 4, "PyTorch", T.PRI)
        p.end()

    def _draw_badge(self, p, x, y, text, color):
        p.setBrush(T.qc(color, 40))
        p.setPen(QPen(T.qc(color), 1))
        p.drawRoundedRect(x, y, 68, 18, 4, 4)
        f = p.font(); f.setPixelSize(9); f.setBold(True); p.setFont(f)
        p.setPen(T.qc(color))
        p.drawText(x + 4, y + 14, text)

    def _draw_rounded_rect(self, p, x, y, w, h, r, fill_color, border_color=None):
        p.setBrush(T.qc(fill_color))
        p.setPen(QPen(T.qc(border_color), 1) if border_color else Qt.PenStyle.NoPen)
        p.drawRoundedRect(x, y, w, h, r, r)

    def _draw_text(self, p, x, y, text, color=T.TXT, size=10, bold=False):
        f = p.font(); f.setPixelSize(size); f.setBold(bold); p.setFont(f)
        p.setPen(T.qc(color)); p.drawText(x, y, text)

    def _draw_arrow(self, p, x1, y1, x2, y2, color=T.OVR):
        p.setPen(QPen(T.qc(color), 2))
        p.drawLine(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        p.drawLine(x2, y2, x2 - 7 * math.cos(angle - 0.4), y2 - 7 * math.sin(angle - 0.4))
        p.drawLine(x2, y2, x2 - 7 * math.cos(angle + 0.4), y2 - 7 * math.sin(angle + 0.4))

    def _draw_neurons(self, p, cx, cy, n, r, spacing, color):
        for i in range(n):
            y = cy - spacing * (n - 1) / 2 + i * spacing
            p.setBrush(T.qc(color, 50))
            p.setPen(QPen(T.qc(color), 1.5))
            p.drawEllipse(QPointF(cx, y), r, r)

    def _draw_arc(self, p, x1, y1, x2, y2, curve_height=18):
        path = QPainterPath()
        path.moveTo(x1, y1)
        mid_x = (x1 + x2) / 2
        path.cubicTo(mid_x, y1 - curve_height, mid_x, y2 - curve_height, x2, y2)
        p.drawPath(path)

    def _draw_mlp(self, p, w, h):
        layers = self.layers or [2, 64, 32, 2]
        n_layers = len(layers)
        layer_x = [w * 0.08 + i * (w * 0.84) / (n_layers - 1) for i in range(n_layers)]
        neuron_r = min(10, max(3, h * 0.04))
        spacing = min(14, (h - 30) / max(layers))
        colors = [T.PRI] + [T.GRN] * (n_layers - 2) + [T.PEA]
        
        for li in range(n_layers - 1):
            for yi in range(layers[li]):
                y1 = h // 2 - spacing * (layers[li] - 1) / 2 + yi * spacing
                for yi2 in range(layers[li + 1]):
                    y2 = h // 2 - spacing * (layers[li + 1] - 1) / 2 + yi2 * spacing
                    p.setPen(QPen(T.qc(T.OVR, 40), 0.5))
                    p.drawLine(int(layer_x[li] + neuron_r), int(y1), int(layer_x[li + 1] - neuron_r), int(y2))
        
        for li, (n, x, c) in enumerate(zip(layers, layer_x, colors)):
            self._draw_neurons(p, x, h // 2, n, neuron_r, spacing, c)
            label = "Input" if li == 0 else "Output" if li == n_layers - 1 else f"Hidden {li}"
            self._draw_text(p, x - 15, h - 8, label, c, 8, True)
        self._draw_text(p, 8, 12, "MLP - Multi-Layer Perceptron", T.PRI, 11, True)
        self._draw_text(p, 8, 24, " → ".join(map(str, layers)), T.DIM, 8)

    def _draw_cnn(self, p, w, h):
        block_w, block_h, y = 75, 46, h // 2 - 23
        positions = [w * .04, w * .18, w * .29, w * .43, w * .54, w * .71, w * .82]
        colors = [T.PRI, T.GRN, T.DIM, T.GRN, T.DIM, T.MAU, T.PEA]
        labels = ["Input\n1×32", "Conv1D\n16ch", "Pool", "Conv1D\n32ch", "Pool", "FC 64", "Out 3"]
        for x, c, label in zip(positions, colors, labels):
            self._draw_rounded_rect(p, x, y, block_w, block_h, 6, c)
            for j, line in enumerate(label.split("\n")):
                self._draw_text(p, x + block_w // 2 - len(line) * 3, y + block_h // 2 - 3 + j * 12, line, T.TXT, 8, j == 0)
        for i in range(len(positions) - 1):
            self._draw_arrow(p, int(positions[i] + block_w), int(y + block_h // 2), int(positions[i + 1]), int(y + block_h // 2))
        self._draw_text(p, 8, 12, "CNN - 1D Convolutional Network", T.PRI, 11, True)

    def _draw_rnn(self, p, w, h):
        n_steps, block_w, block_h, cy = 4, 68, 42, h // 2
        gap = (w - 70) / (n_steps + 1)
        for i in range(n_steps):
            x = 45 + i * gap
            self._draw_rounded_rect(p, x, cy - block_h // 2, block_w, block_h, 6, T.GRN)
            self._draw_text(p, x + block_w // 2 - 10, cy - 2, "RNN", T.TXT, 9, True)
            self._draw_text(p, x + block_w // 2 - 8, cy + 11, f"h{i}", T.MAU, 7)
            if i < n_steps - 1:
                self._draw_arrow(p, int(x + block_w), int(cy - 8), int(x + gap), int(cy - 8), T.PRI)
                self._draw_arrow(p, int(x + block_w), int(cy + 8), int(x + gap), int(cy + 8), T.MAU)
            if i > 0:
                p.setPen(QPen(T.qc(T.MAU, 90), 1.5, Qt.PenStyle.DashLine))
                self._draw_arc(p, x - 6, cy + 4, x - 6 - gap + 14, cy + 4)
            self._draw_text(p, x + block_w // 2 - 4, cy - block_h // 2 - 11, f"x{i}", T.PRI, 7)
            self._draw_text(p, x + block_w // 2 - 4, cy + block_h // 2 + 11, f"y{i}", T.PEA, 7)
        self._draw_text(p, 8, 12, "RNN - Recurrent Neural Network (unrolled)", T.PRI, 11, True)

    def _draw_lstm(self, p, w, h):
        n_steps, block_w, block_h, cy = 3, 88, 54, h // 2
        gap = (w - 90) / (n_steps + 1)
        gate_colors = [T.RED, T.YEL, T.TEAL, T.RED]
        gate_names = ["f", "i", "C~", "o"]
        for i in range(n_steps):
            x = 55 + i * gap
            gate_w, gate_gap = 17, (block_w - 4 * gate_w) / 5
            self._draw_rounded_rect(p, x, cy - block_h // 2, block_w, block_h, 6, T.S2, T.MAU)
            for gi, (gc, gn) in enumerate(zip(gate_colors, gate_names)):
                gx = x + gate_gap + gi * (gate_w + gate_gap)
                self._draw_rounded_rect(p, gx, cy - block_h // 2 + 5, gate_w, 11, 3, gc)
                self._draw_text(p, gx + gate_w // 2 - 2, cy - block_h // 2 + 14, gn, T.BG, 7, True)
            self._draw_text(p, x + block_w // 2 - 8, cy + 2, "LSTM", T.TXT, 9, True)
            if i < n_steps - 1:
                self._draw_arrow(p, int(x + block_w), int(cy - 11), int(x + gap), int(cy - 11), T.PRI)
                self._draw_arrow(p, int(x + block_w), int(cy + 11), int(x + gap), int(cy + 11), T.MAU)
                p.setPen(QPen(T.qc(T.TEAL, 80), 1.5, Qt.PenStyle.DashLine))
                self._draw_arc(p, x - 4, cy + 15, x - 4 - gap + 12, cy + 15, 18)
        self._draw_text(p, 8, 12, "LSTM - Long Short-Term Memory (unrolled)", T.PRI, 11, True)

    def _draw_transformer(self, p, w, h):
        block_w, block_h, cy, gap = 82, 34, h // 2, 7
        blocks = [("Input", T.PRI), ("PosEnc", T.DIM), ("MultiHead\nAttn", T.YEL),
                  ("Add&Norm", T.TEAL), ("FFN", T.GRN), ("Add&Norm", T.TEAL), ("Output", T.PEA)]
        total_w = len(blocks) * (block_w + gap) - gap
        start_x = (w - total_w) // 2
        positions = []
        for i, (name, color) in enumerate(blocks):
            x = start_x + i * (block_w + gap)
            positions.append(x)
            self._draw_rounded_rect(p, x, cy - block_h // 2, block_w, block_h, 6, color)
            for j, line in enumerate(name.split("\n")):
                self._draw_text(p, x + block_w // 2 - len(line) * 3, cy - 1 + j * 11, line, T.TXT, 8, j == 0)
            if i < len(blocks) - 1:
                self._draw_arrow(p, int(x + block_w), int(cy), int(positions[i + 1]), int(cy))
        residual_y = cy + block_h // 2 + 5
        p.setPen(QPen(T.qc(T.TEAL, 70), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(int(positions[2]), int(residual_y), int(positions[4]), int(residual_y))
        self._draw_text(p, 8, 12, "Transformer - Self-Attention Encoder", T.PRI, 11, True)

    def _draw_gan(self, p, w, h):
        mid_y = h // 2
        g_x, d_x = w * 0.05, w * 0.55
        block_w, block_h = w * 0.38, 50
        self._draw_rounded_rect(p, g_x, mid_y - block_h // 2, block_w, block_h, 8, T.GRN)
        self._draw_text(p, g_x + block_w // 2 - 40, mid_y - 5, "Generator (G)", T.BG, 11, True)
        self._draw_text(p, g_x + block_w // 2 - 55, mid_y + 10, "z → FC(64) → FC(64) → x", T.BG, 8)
        self._draw_rounded_rect(p, d_x, mid_y - block_h // 2, block_w, block_h, 8, T.RED)
        self._draw_text(p, d_x + block_w // 2 - 55, mid_y - 5, "Discriminator (D)", T.BG, 11, True)
        self._draw_text(p, d_x + block_w // 2 - 60, mid_y + 10, "x → FC(64) → FC(64) → [0,1]", T.BG, 8)
        self._draw_arrow(p, int(g_x + block_w), int(mid_y - 8), int(d_x), int(mid_y - 8), T.PRI)
        self._draw_text(p, int((g_x + block_w + d_x) / 2) - 15, int(mid_y - 14), "fake x", T.PRI, 8)
        p.setPen(QPen(T.qc(T.RED, 80), 1.5, Qt.PenStyle.DashLine))
        self._draw_arc(p, int(d_x + block_w // 2), int(mid_y - block_h // 2), int(g_x + block_w // 2), int(mid_y - block_h // 2), 25)
        self._draw_text(p, int((g_x + block_w + d_x) / 2) - 15, int(mid_y - 30), "∇ to G", T.RED, 8)
        self._draw_text(p, 8, 12, "GAN - Generative Adversarial Network", T.PRI, 11, True)


class InfoPanel(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
    def show_architecture_info(self, arch_type: str):
        self.setHtml(ARCHITECTURE_DESCRIPTIONS.get(arch_type, ""))
        
    def show_dataset_info(self, dataset_name: str):
        self.setHtml(DATASET_DESCRIPTIONS.get(dataset_name, ""))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Training Visualizer")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        self._training = False
        self._thread = None
        self._tip_index = 0
        self._tip_timer = QTimer()
        self._tip_timer.timeout.connect(self._rotate_tip)
        
        self._build_ui()
        self._connect_signals()
        self._on_arch_changed(self.arch_combo.currentText())
        self._update_ui_state()
        
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 1000])
        
        self.statusBar().showMessage("Ready — Select an architecture and dataset to begin")
        
    def _build_left_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        
        self.arch_widget = ArchWidget()
        layout.addWidget(self.arch_widget)
        
        self.info_panel = InfoPanel()
        layout.addWidget(self.info_panel)
        
        layout.addWidget(self._build_arch_group())
        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_hyperparam_group())
        layout.addWidget(self._build_control_group())
        
        self.tip_label = QLabel(TRAINING_TIPS["mlp"][0])
        self.tip_label.setObjectName("tip")
        self.tip_label.setWordWrap(True)
        self.tip_label.setMinimumHeight(40)
        layout.addWidget(self.tip_label)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll
        
    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(4)
        
        self.tabs = QTabWidget()
        
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        self.output_plot = OutputPlot()
        output_layout.addWidget(self.output_plot)
        self.tabs.addTab(output_tab, "Output")
        
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        self.conf_plot = ConfusionMatrixPlot()
        self.feat_plot = FeatureSpacePlot()
        analysis_layout.addWidget(self.conf_plot)
        analysis_layout.addWidget(self.feat_plot)
        self.tabs.addTab(analysis_tab, "Analysis")
        
        training_tab = QWidget()
        training_layout = QVBoxLayout(training_tab)
        self.loss_plot = LossPlot()
        self.metric_plot = MetricPlot()
        self.lr_plot = LRPlot()
        self.grad_plot = GradFlowPlot()
        self.weight_plot = WeightDistributionPlot()
        training_layout.addWidget(self.loss_plot)
        training_layout.addWidget(self.metric_plot)
        training_layout.addWidget(self.lr_plot)
        training_layout.addWidget(self.grad_plot)
        training_layout.addWidget(self.weight_plot)
        self.tabs.addTab(training_tab, "Training")
        
        layout.addWidget(self.tabs)
        return container
        
    def _build_arch_group(self) -> QGroupBox:
        group = QGroupBox("Network Architecture")
        layout = QFormLayout(group)
        layout.setSpacing(6)
        
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["mlp", "cnn", "rnn", "lstm", "transformer", "gan"])
        self.arch_combo.setToolTip("""<b>Select Architecture</b><br><br>
Choose which neural network type to train.<br><br>
<b>MLP:</b> General purpose, good for 2D classification<br>
<b>CNN:</b> Best for signal/waveform patterns<br>
<b>RNN/LSTM:</b> For sequence prediction tasks<br>
<b>Transformer:</b> Modern attention-based sequence model<br>
<b>GAN:</b> Learn to generate data from noise""")
        layout.addRow("Architecture:", self.arch_combo)
        
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        self.optimizer_combo.setToolTip(OPTIMIZER_TOOLTIPS["Adam"])
        layout.addRow("Optimizer:", self.optimizer_combo)
        
        self.param_label = QLabel("Parameters: —")
        self.param_label.setObjectName("info")
        layout.addRow(self.param_label)
        
        return group
        
    def _build_data_group(self) -> QGroupBox:
        group = QGroupBox("Dataset")
        layout = QFormLayout(group)
        layout.setSpacing(6)
        
        self.data_combo = QComboBox()
        self.data_combo.setToolTip("""<b>Select Dataset</b><br><br>
Available datasets depend on the selected architecture.<br><br>
Each dataset highlights different learning challenges:
• Simpler datasets → see basics
• Harder datasets → test network capacity""")
        layout.addRow("Dataset:", self.data_combo)
        
        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.0, 0.5)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setDecimals(2)
        self.test_split_spin.setValue(0.2)
        self.test_split_spin.setSuffix("  (test ratio)")
        self.test_split_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["test_split"])
        layout.addRow("Test Split:", self.test_split_spin)
        
        self.split_info_label = QLabel("")
        self.split_info_label.setObjectName("desc")
        self.split_info_label.setWordWrap(True)
        layout.addRow(self.split_info_label)
        
        self.data_info_label = QLabel("")
        self.data_info_label.setObjectName("desc")
        self.data_info_label.setWordWrap(True)
        layout.addRow(self.data_info_label)
        
        return group
        
    def _build_hyperparam_group(self) -> QGroupBox:
        group = QGroupBox("Hyperparameters")
        group.setToolTip("""<b>Hyperparameters</b><br><br>
Settings that control <i>how</i> the network learns.
Unlike model parameters (weights), these are set before training
and affect the training process itself.""")
        layout = QFormLayout(group)
        layout.setSpacing(6)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["learning_rate"])
        layout.addRow("Learning Rate:", self.lr_spin)
        
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 0.1)
        self.wd_spin.setSingleStep(0.001)
        self.wd_spin.setDecimals(4)
        self.wd_spin.setValue(0.0)
        self.wd_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["weight_decay"])
        layout.addRow("Weight Decay:", self.wd_spin)
        
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["dropout"])
        layout.addRow("Dropout:", self.dropout_spin)
        
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 5000)
        self.epoch_spin.setSingleStep(50)
        self.epoch_spin.setValue(200)
        self.epoch_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["epochs"])
        layout.addRow("Epochs:", self.epoch_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 256)
        self.batch_spin.setSingleStep(8)
        self.batch_spin.setValue(32)
        self.batch_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["batch_size"])
        layout.addRow("Batch Size:", self.batch_spin)
        
        return group
        
    def _build_control_group(self) -> QGroupBox:
        group = QGroupBox("Training Controls")
        layout = QHBoxLayout(group)
        
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.setToolTip("""<b>Start Training</b><br><br>
Begins the training loop. The network will learn from the training set
and you can watch performance on both train and test sets in real-time.""")
        layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏸ Pause")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip("""<b>Pause Training</b><br><br>
Pauses training at the current epoch.
Click Resume to continue from where you left off.""")
        layout.addWidget(self.stop_btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.setToolTip("""<b>Reset Everything</b><br><br>
Stops training and resets all plots and model weights.
Use this to start fresh with different settings.""")
        layout.addWidget(self.reset_btn)
        
        return group
        
    def _connect_signals(self):
        self.arch_combo.currentTextChanged.connect(self._on_arch_changed)
        self.data_combo.currentTextChanged.connect(self._on_data_changed)
        self.optimizer_combo.currentTextChanged.connect(self._on_optimizer_changed)
        self.test_split_spin.valueChanged.connect(self._on_split_changed)
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn.clicked.connect(self._on_stop)
        self.reset_btn.clicked.connect(self._on_reset)
        
    def _on_arch_changed(self, arch: str):
        datasets_map = {
            "mlp": list(MLP_DATASETS.keys()),
            "cnn": list(CNN_DATASETS.keys()),
            "rnn": list(SEQ_DATASETS.keys()),
            "lstm": list(SEQ_DATASETS.keys()),
            "transformer": list(SEQ_DATASETS.keys()),
            "gan": GAN_MODES,
        }
        
        self.data_combo.blockSignals(True)
        self.data_combo.clear()
        self.data_combo.addItems(datasets_map.get(arch, []))
        self.data_combo.blockSignals(False)
        
        layers = ARCHITECTURE_INFO.get(arch, {}).get("default_layers")
        self.arch_widget.set_type(arch, layers)
        self.info_panel.show_architecture_info(arch)
        
        is_gan = arch == "gan"
        self.test_split_spin.setEnabled(not is_gan)
        if is_gan:
            self.test_split_spin.setToolTip("GANs don't use traditional train/test splits")
        else:
            self.test_split_spin.setToolTip(HYPERPARAMETER_TOOLTIPS["test_split"])
        
        try:
            model = make_model(arch)
            if arch == "gan":
                params = count_parameters(model[0]) + count_parameters(model[1])
                self.param_label.setText(f"Parameters: G={count_parameters(model[0]):,} + D={count_parameters(model[1]):,} = {params:,}")
            else:
                self.param_label.setText(f"Parameters: {count_parameters(model):,}")
        except Exception:
            self.param_label.setText("Parameters: —")
            
        self._on_data_changed(self.data_combo.currentText())
        self._on_split_changed(self.test_split_spin.value())
        
    def _on_data_changed(self, dataset: str):
        self.data_info_label.setText(DATASET_DESCRIPTIONS.get(dataset, "")[:80] + "...")
        self.data_info_label.setToolTip(DATASET_DESCRIPTIONS.get(dataset, ""))
        
    def _on_split_changed(self, value: float):
        pct = int((1 - value) * 100)
        test_pct = int(value * 100)
        self.split_info_label.setText(f"📊 {pct}% train, {test_pct}% test — model never sees test data during training")
        
    def _on_optimizer_changed(self, opt: str):
        self.optimizer_combo.setToolTip(OPTIMIZER_TOOLTIPS.get(opt, ""))
        
    def _on_start(self):
        if self._training and self._thread and self._thread._paused:
            self._thread.resume()
            self.stop_btn.setText("⏸ Pause")
            self.statusBar().showMessage("Training resumed")
            return
            
        arch = self.arch_combo.currentText()
        dataset = self.data_combo.currentText()
        
        if not dataset:
            self.statusBar().showMessage("Please select a dataset")
            return
            
        self._prepare_data(arch, dataset)
        self._prepare_model(arch)
        self._start_training()
        
    def _prepare_data(self, arch: str, dataset: str):
        test_ratio = self.test_split_spin.value()
        
        if arch == "gan":
            self._X_train = self._Y_train = None
            self._X_test = self._Y_test = None
            self._gd = make_gan_data(500, dataset)
            return
        
        if arch == "mlp":
            X, y = MLP_DATASETS[dataset]()
        elif arch == "cnn":
            X, y = CNN_DATASETS[dataset]()
        else:
            X, y = SEQ_DATASETS[dataset]()
        
        if test_ratio > 0:
            self._X_train, self._X_test, self._Y_train, self._Y_test = train_test_split(
                X, y, test_ratio=test_ratio, stratify=(y.ndim == 1)
            )
        else:
            self._X_train, self._Y_train = X, y
            self._X_test = self._Y_test = None
        
        self._gd = None
        
    def _prepare_model(self, arch: str):
        dropout = self.dropout_spin.value()
        if arch == "mlp":
            layers = ARCHITECTURE_INFO["mlp"]["default_layers"]
            self._model = MLP(layers, dropout)
            self.arch_widget.set_type(arch, layers)
        elif arch == "cnn":
            self._model = CNN(dropout)
        elif arch == "rnn":
            self._model = RNN()
        elif arch == "lstm":
            self._model = LSTM()
        elif arch == "transformer":
            self._model = Transformer()
        elif arch == "gan":
            self._model = (Generator(), Discriminator())
            
    def _start_training(self):
        arch = self.arch_combo.currentText()
        
        self._thread = TrainThread()
        self._thread.setup(
            ntype=arch,
            model=self._model,
            X_train=self._X_train,
            Y_train=self._Y_train,
            X_test=self._X_test,
            Y_test=self._Y_test,
            gd=self._gd,
            lr=self.lr_spin.value(),
            wd=self.wd_spin.value(),
            bs=self.batch_spin.value(),
            epochs=self.epoch_spin.value(),
            optimizer=self.optimizer_combo.currentText(),
        )
        self._thread.epoch_sig.connect(self._on_epoch)
        self._thread.finished_sig.connect(self._on_finished)
        
        self._training = True
        self._tip_index = 0
        self._tip_timer.start(4000)
        self._update_ui_state()
        
        has_test = self._X_test is not None
        test_info = f" | Test set: {len(self._X_test)} samples" if has_test else " | No test split"
        self.statusBar().showMessage(f"Training {arch.upper()} started... | Train: {len(self._X_train)} samples{test_info}")
        self._thread.start()
        
    def _on_epoch(self, epoch: int, train_loss: float, train_metric: float, 
                  test_loss: float, test_metric: float, weight_info: dict, output_data: dict):
        arch = self.arch_combo.currentText()
        
        self.loss_plot.update_plot(epoch, train_loss, test_loss)
        self.metric_plot.update_plot(epoch, train_metric, test_metric, arch)
        self.lr_plot.update_plot(epoch, weight_info.get("lr", self.lr_spin.value()))
        self.grad_plot.update_plot(weight_info.get("grad_norms", {}))
        self.weight_plot.update_plot(weight_info)
        
        if output_data.get("type") == "boundary":
            self.output_plot.update_plot(output_data)
            if output_data.get("confusion"):
                self.conf_plot.update_plot(output_data["confusion"])
            if output_data.get("features"):
                self.feat_plot.update_plot(output_data["features"])
        elif output_data.get("type") == "seq":
            self.output_plot.update_plot(output_data)
        elif output_data.get("type") == "gan":
            self.output_plot.update_plot(output_data)
            
        metric_name = "Accuracy" if arch in ("mlp", "cnn") else "R²" if arch != "gan" else "G Score"
        status = f"Epoch {epoch}/{self.epoch_spin.value()} | Train {metric_name}: {train_metric:.4f}"
        
        if test_metric is not None:
            gap = train_metric - test_metric
            gap_str = f" | Gap: {gap:+.4f}"
            if gap > 0.1 and arch in ("mlp", "cnn"):
                gap_str += " ⚠️ overfitting"
            status += f" | Test {metric_name}: {test_metric:.4f}{gap_str}"
        
        self.statusBar().showMessage(status)
        
    def _on_finished(self):
        self._training = False
        self._tip_timer.stop()
        self._update_ui_state()
        self.statusBar().showMessage("Training complete")
        
    def _on_stop(self):
        if self._thread:
            if self._thread._paused:
                self._thread.resume()
                self.stop_btn.setText("⏸ Pause")
                self.statusBar().showMessage("Training resumed")
            else:
                self._thread.pause()
                self.stop_btn.setText("▶ Resume")
                self.statusBar().showMessage("Training paused")
                
    def _on_reset(self):
        if self._thread:
            self._thread.stop()
            self._thread.wait(1000)
        self._training = False
        self._tip_timer.stop()
        
        self.loss_plot.clear_data()
        self.metric_plot.clear_data()
        self.lr_plot.clear_data()
        self.loss_plot.clear_all()
        self.metric_plot.clear_all()
        self.lr_plot.clear_all()
        self.grad_plot.clear_all()
        self.output_plot.clear_all()
        self.conf_plot.clear_all()
        self.feat_plot.clear_all()
        self.weight_plot.clear_all()
        
        self.stop_btn.setText("⏸ Pause")
        self.tip_label.setText(TRAINING_TIPS[self.arch_combo.currentText()][0])
        self._update_ui_state()
        self.statusBar().showMessage("Reset complete — Ready to train")
        
    def _rotate_tip(self):
        arch = self.arch_combo.currentText()
        tips = TRAINING_TIPS.get(arch, [])
        if tips:
            self._tip_index = (self._tip_index + 1) % len(tips)
            self.tip_label.setText(tips[self._tip_index])
            
    def _update_ui_state(self):
        training = self._training
        self.start_btn.setEnabled(not training)
        self.stop_btn.setEnabled(training)
        self.arch_combo.setEnabled(not training)
        self.data_combo.setEnabled(not training)
        self.lr_spin.setEnabled(not training)
        self.wd_spin.setEnabled(not training)
        self.dropout_spin.setEnabled(not training)
        self.epoch_spin.setEnabled(not training)
        self.batch_spin.setEnabled(not training)
        self.optimizer_combo.setEnabled(not training)
        self.test_split_spin.setEnabled(not training and self.arch_combo.currentText() != "gan")

    def closeEvent(self, event):
        """Clean up the training thread before the window closes."""
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(1000)
        event.accept()