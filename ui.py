"""All UI widgets: architecture diagram, plot canvases, and main window."""

import math
import csv
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QLineEdit, QCheckBox, QSlider, QTabWidget, QTableWidget, QHeaderView,
    QAbstractItemView, QScrollArea, QFileDialog, QApplication, QSizePolicy,
)
from PySide6.QtCore import Qt, QDateTime, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath, QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("QtAgg")

from theme import T
from factory import make_model, count_params
from models_tensorflow import TF_AVAILABLE
from datasets import MLP_DATASETS, make_signals, make_sine, make_gan_data
from training import TrainThread

# ── Description text constants ──────────────────────────────────────────
DESCS = {
    "mlp": "MLP: Fully connected layers with ReLU. Compare PT nn.Linear vs TF Dense — same math, different APIs.",
    "cnn": "CNN: Conv1D filters detect local patterns. Compare PT Conv1d vs TF Conv1D — shared weights learn edges/waves.",
    "rnn": "RNN: h_t = tanh(W_h·h_{t-1} + W_x·x_t). Limited by vanishing gradients. Compare PT nn.RNN vs TF SimpleRNN.",
    "lstm": "LSTM: Cell state C_t + gates (f, i, ĩ, o). Solves vanishing gradient. Compare PT nn.LSTM vs TF LSTM.",
    "transformer": "Transformer: softmax(QK^T/√d)V — all positions in parallel. Architecture behind GPT, BERT, etc.",
    "gan": "GAN: G(z) generates fakes, D(x) scores real vs fake. Adversarial min-max game. Compare PT vs TF loops.",
}
DATA_INFO = {
    "mlp": "Binary classification on 2D data\nSelect dataset below",
    "cnn": "1D signal classification\n500×32, 3 wave classes",
    "rnn": "Next-step sine prediction\n300 seq × 30 steps",
    "lstm": "Next-step sine prediction\n300 seq × 30 steps",
    "transformer": "Next-step prediction via attention\n300 seq × 30 steps",
    "gan": "Learn 2D distribution from real data\nSelect target shape below",
}


# ═══════════════════════════════════════════════════════════════════════
#  Architecture Widget
# ═══════════════════════════════════════════════════════════════════════
class ArchWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ntype = "mlp"
        self.fw = "pytorch"
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_type(self, t, fw=None, layers=None):
        self.ntype = t
        self.fw = fw or self.fw
        self.layers = layers
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), T.qc(T.S1))
        getattr(self, f"_draw_{self.ntype}")(p, self.width(), self.height())
        self._draw_badge(p)
        p.end()

    # ── primitives ──────────────────────────────────────────────────────
    def _rr(self, p, x, y, w, h, r, col, border=None):
        p.setBrush(T.qc(col))
        p.setPen(QPen(T.qc(border), 1) if border else Qt.PenStyle.NoPen)
        p.drawRoundedRect(x, y, w, h, r, r)

    def _tx(self, p, x, y, t, c=T.TXT, sz=10, b=False):
        f = p.font(); f.setPixelSize(sz); f.setBold(b); p.setFont(f)
        p.setPen(T.qc(c)); p.drawText(x, y, t)

    def _ar(self, p, x1, y1, x2, y2, c=T.OVR):
        p.setPen(QPen(T.qc(c), 2)); p.drawLine(x1, y1, x2, y2)
        a = math.atan2(y2 - y1, x2 - x1); al = 7
        p.drawLine(x2, y2, x2 - al * math.cos(a - .4), y2 - al * math.sin(a - .4))
        p.drawLine(x2, y2, x2 - al * math.cos(a + .4), y2 - al * math.sin(a + .4))

    def _neurons(self, p, cx, cy, n, r, sp, c):
        ys = [cy - sp * (n - 1) / 2 + i * sp for i in range(n)]
        for yy in ys:
            p.setBrush(T.qc(c, 50)); p.setPen(QPen(T.qc(c), 1.5))
            p.drawEllipse(QPointF(cx, yy), r, r)

    def _arc(self, p, x1, y1, x2, y2, curv=18):
        path = QPainterPath(); path.moveTo(x1, y1); mx = (x1 + x2) / 2
        path.cubicTo(mx, y1 - curv, mx, y2 - curv, x2, y2); p.drawPath(path)

    def _draw_badge(self, p):
        txt = "PyTorch" if self.fw == "pytorch" else "TensorFlow"
        bc = T.PRI if self.fw == "pytorch" else T.PEA
        self._rr(p, self.width() - 78, 4, 74, 18, 4, T.qc(bc, 40), bc)
        self._tx(p, self.width() - 75, 17, txt, bc, 9, True)

    # ── architecture drawings ───────────────────────────────────────────
    def _draw_mlp(self, p, w, h):
        ls = self.layers or [2, 64, 32, 2]; nl = len(ls)
        lx = [w * 0.08 + i * (w * 0.84) / (nl - 1) for i in range(nl)]
        nr = min(10, max(3, h * 0.04)); sp = min(14, (h - 30) / max(ls))
        cs = [T.PRI] + [T.GRN] * (nl - 2) + [T.PEA]
        for li, (n, x, c) in enumerate(zip(ls, lx, cs)):
            self._neurons(p, x, h // 2, n, nr, sp, c)
            lb = "Input" if li == 0 else "Output" if li == nl - 1 else f"H{li}"
            self._tx(p, x - 10, h - 8, lb, c, 8, True)
        for li in range(nl - 1):
            for yi in range(ls[li]):
                y1 = h // 2 - sp * (ls[li] - 1) / 2 + yi * sp
                for yi2 in range(ls[li + 1]):
                    y2 = h // 2 - sp * (ls[li + 1] - 1) / 2 + yi2 * sp
                    p.setPen(QPen(T.qc(T.OVR, 40), 0.5))
                    p.drawLine(int(lx[li] + nr), int(y1), int(lx[li + 1] - nr), int(y2))
        self._tx(p, 8, 12, "MLP — Multi-Layer Perceptron", T.PRI, 11, True)
        self._tx(p, 8, 24, " → ".join(map(str, ls)), T.DIM, 8)

    def _draw_cnn(self, p, w, h):
        bw, bh, y = 75, 46, h // 2 - 23
        xs = [w * .04, w * .18, w * .29, w * .43, w * .54, w * .71, w * .82]
        cs = [T.PRI, T.GRN, T.DIM, T.GRN, T.DIM, T.MAU, T.PEA]
        ns = ["Input\n1×32", "Conv1D\n16ch", "Pool", "Conv1D\n32ch", "Pool", "FC 64", "Out 3"]
        for x, c, nm in zip(xs, cs, ns):
            self._rr(p, x, y, bw, bh, 6, c)
            for j, ln in enumerate(nm.split("\n")):
                self._tx(p, x + bw // 2 - len(ln) * 3, y + bh // 2 - 3 + j * 12, ln, T.TXT, 8, j == 0)
        for i in range(len(xs) - 1):
            self._ar(p, int(xs[i] + bw), int(y + bh // 2), int(xs[i + 1]), int(y + bh // 2))
        self._tx(p, 8, 12, "CNN — 1D Convolutional Network", T.PRI, 11, True)

    def _draw_rnn(self, p, w, h):
        ns, bw, bh, cy = 4, 68, 42, h // 2
        gap = (w - 70) / (ns + 1)
        for i in range(ns):
            x = 45 + i * gap; self._rr(p, x, cy - bh // 2, bw, bh, 6, T.GRN)
            self._tx(p, x + bw // 2 - 10, cy - 2, "RNN", T.TXT, 9, True)
            self._tx(p, x + bw // 2 - 8, cy + 11, f"h{i}", T.MAU, 7)
            if i < ns - 1:
                self._ar(p, int(x + bw), int(cy - 8), int(x + gap), int(cy - 8), T.PRI)
                self._ar(p, int(x + bw), int(cy + 8), int(x + gap), int(cy + 8), T.MAU)
            if i > 0:
                p.setPen(QPen(T.qc(T.MAU, 90), 1.5, Qt.PenStyle.DashLine))
                self._arc(p, x - 6, cy + 4, x - 6 - gap + 14, cy + 4)
            self._tx(p, x + bw // 2 - 4, cy - bh // 2 - 11, f"x{i}", T.PRI, 7)
            self._tx(p, x + bw // 2 - 4, cy + bh // 2 + 11, f"y{i}", T.PEA, 7)
        self._tx(p, 8, 12, "RNN — Recurrent Neural Network (unrolled)", T.PRI, 11, True)

    def _draw_lstm(self, p, w, h):
        ns, bw, bh, cy = 3, 88, 54, h // 2
        gap = (w - 90) / (ns + 1)
        gc, gn = [T.RED, T.YEL, T.TEAL, T.RED], ["f", "i", "ĩ", "o"]
        for i in range(ns):
            x = 55 + i * gap; gw, gg = 17, (bw - 4 * gw) / 5
            self._rr(p, x, cy - bh // 2, bw, bh, 6, T.S2, T.MAU)
            for gi, (c2, n2) in enumerate(zip(gc, gn)):
                gx = x + gg + gi * (gw + gg)
                self._rr(p, gx, cy - bh // 2 + 5, gw, 11, 3, c2)
                self._tx(p, gx + gw // 2 - 2, cy - bh // 2 + 14, n2, T.BG, 7, True)
            self._tx(p, x + bw // 2 - 8, cy + 2, "LSTM", T.TXT, 9, True)
            if i < ns - 1:
                self._ar(p, int(x + bw), int(cy - 11), int(x + gap), int(cy - 11), T.PRI)
                self._ar(p, int(x + bw), int(cy + 11), int(x + gap), int(cy + 11), T.MAU)
                p.setPen(QPen(T.qc(T.TEAL, 80), 1.5, Qt.PenStyle.DashLine))
                self._arc(p, x - 4, cy + 15, x - 4 - gap + 12, cy + 15, 18)
        self._tx(p, 8, 12, "LSTM — Long Short-Term Memory (unrolled)", T.PRI, 11, True)

    def _draw_transformer(self, p, w, h):
        bw, bh, cy, gap = 82, 34, h // 2, 7
        blks = [("Input", T.PRI), ("PosEnc", T.DIM), ("MultiHead\nAttn", T.YEL),
                ("Add&Norm", T.TEAL), ("FFN", T.GRN), ("Add&Norm", T.TEAL), ("Output", T.PEA)]
        tw = len(blks) * (bw + gap) - gap; sx = (w - tw) // 2; xs = []
        for i, (nm, c) in enumerate(blks):
            x = sx + i * (bw + gap); xs.append(x)
            self._rr(p, x, cy - bh // 2, bw, bh, 6, c)
            for j, ln in enumerate(nm.split("\n")):
                self._tx(p, x + bw // 2 - len(ln) * 3, cy - 1 + j * 11, ln, T.TXT, 8, j == 0)
            if i < len(blks) - 1:
                self._ar(p, int(x + bw), int(cy), int(xs[i + 1]), int(cy))
        ry = cy + bh // 2 + 5
        p.setPen(QPen(T.qc(T.TEAL, 70), 1.5, Qt.PenStyle.DashLine))
        p.drawLine(int(xs[2]), int(ry), int(xs[4]), int(ry))
        self._tx(p, 8, 12, "Transformer — Encoder with Self-Attention", T.PRI, 11, True)

    def _draw_gan(self, p, w, h):
        by1, by2, bw, bh = h // 2 - 48, h // 2 + 18, 72, 32
        sx, gg = w * 0.06, (w * 0.40) / 4
        gn2 = [("z~N(0,1)", T.DIM), ("FC 64", T.GRN), ("FC 64", T.GRN), ("FC 2", T.PEA)]
        dn = [("Data x", T.PRI), ("FC 64", T.RED), ("FC 64", T.RED), ("σ", T.RED)]
        for i, (nm, c) in enumerate(gn2):
            x = sx + i * gg; self._rr(p, x, by1, bw, bh, 6, c)
            self._tx(p, x + bw // 2 - len(nm) * 3, by1 + bh // 2 + 3, nm, T.TXT, 8)
            if i < 3: self._ar(p, int(x + bw), int(by1 + bh // 2), int(x + gg), int(by1 + bh // 2))
        for i, (nm, c) in enumerate(dn):
            x = sx + i * gg; self._rr(p, x, by2, bw, bh, 6, c)
            self._tx(p, x + bw // 2 - len(nm) * 3, by2 + bh // 2 + 3, nm, T.TXT, 8)
            if i < 3: self._ar(p, int(x + bw), int(by2 + bh // 2), int(x + gg), int(by2 + bh // 2))
        fx = sx + 3 * gg + bw + 12
        self._rr(p, fx, by1, bw, bh, 6, T.MAU)
        self._tx(p, fx + bw // 2 - 18, by1 + bh // 2 + 3, "Fake Data", T.TXT, 8)
        self._ar(p, int(sx + 3 * gg + bw), int(by1 + bh // 2), int(fx), int(by1 + bh // 2))
        self._ar(p, int(fx + bw // 2), int(by1 + bh), int(fx + bw // 2), int(by2), T.MAU)
        self._ar(p, int(fx), int(by2 + bh // 2), int(sx + gg), int(by2 + bh // 2))
        self._tx(p, w * .50, by1 - 7, "GENERATOR", T.GRN, 10, True)
        self._tx(p, w * .50, by2 - 7, "DISCRIMINATOR", T.RED, 10, True)
        self._tx(p, 8, 12, "GAN — Generative Adversarial Network", T.PRI, 11, True)


# ═══════════════════════════════════════════════════════════════════════
#  Plot Canvas Base + Subclasses
# ═══════════════════════════════════════════════════════════════════════
class PlotCanvas(FigureCanvas):
    def __init__(self, title="", nrows=1, ncols=1):
        self.fig = Figure(figsize=(4, 3), dpi=85)
        self.fig.patch.set_facecolor(T.S1)
        super().__init__(self.fig)
        if nrows == 1 and ncols == 1:
            self.axes = [self.fig.add_subplot(111)]
        else:
            self.axes = self.fig.subplots(nrows, ncols, sharex=False, sharey=False).flatten()
        self._sty(title)
        self.fig.tight_layout(pad=1.0)

    def _sty(self, title=""):
        for ax in self.axes:
            ax.set_facecolor(T.S1)
            ax.tick_params(colors=T.DIM, labelsize=6)
            for s in ax.spines.values():
                s.set_color(T.OVR)
            ax.grid(True, alpha=0.1, color=T.DIM)
        if title:
            self.axes[0].set_title(title, color=T.SUB, fontsize=9, pad=5)

    def clear_all(self):
        self.fig.clear()
        self.axes = [self.fig.add_subplot(111)]
        self._sty()
        self.fig.tight_layout(pad=1.0)


class LossPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Training Loss")
        self.losses = []

    def update_plot(self, ep, l):
        self.losses.append((ep, l))
        self.clear_all()
        self._sty("Training Loss")
        if self.losses:
            e, v = zip(*self.losses)
            self.axes[0].plot(e, v, color=T.PRI, lw=1.2)
            self.axes[0].fill_between(e, v, alpha=0.06, color=T.PRI)
        self.draw_idle()


class MetricPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Metric")
        self.data = []

    def update_plot(self, ep, m, nt=""):
        self.data.append((ep, m))
        self.clear_all()
        lb = {"mlp": "Accuracy", "cnn": "Accuracy", "rnn": "R²",
              "lstm": "R²", "transformer": "R²", "gan": "G Score"}.get(nt, "Metric")
        self._sty(lb)
        if self.data:
            e, v = zip(*self.data)
            c = T.GRN if nt in ("mlp", "cnn") else T.PEA
            self.axes[0].plot(e, v, color=c, lw=1.2)
            self.axes[0].fill_between(e, v, alpha=0.06, color=c)
        self.draw_idle()


class LRPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Learning Rate")
        self.data = []

    def update_plot(self, ep, lr):
        self.data.append((ep, lr))
        self.clear_all()
        self._sty("Learning Rate")
        if self.data:
            e, v = zip(*self.data)
            self.axes[0].plot(e, v, color=T.YEL, lw=1.2)
            self.axes[0].fill_between(e, v, alpha=0.08, color=T.YEL)
        self.draw_idle()


class GradFlowPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Gradient Flow")

    def update_plot(self, gn):
        self.clear_all()
        self._sty("Gradient Flow (log ‖∇‖)")
        if not gn:
            self.draw_idle()
            return
        names = list(gn.keys())
        vals = list(gn.values())
        short = [n.split(".")[-1] if "." in n else n for n in names]
        cols = [T.GRN if v > 1e-4 else (T.YEL if v > 1e-6 else T.RED) for v in vals]
        self.axes[0].barh(range(len(short)), vals, color=cols, alpha=0.8)
        self.axes[0].set_yticks(range(len(short)))
        self.axes[0].set_yticklabels(short, fontsize=5)
        self.axes[0].set_xscale("log")
        self.axes[0].axvline(x=1e-6, color=T.RED, ls="--", alpha=0.4, lw=0.7)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class OutputPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Network Output")

    def update_plot(self, od):
        self.clear_all()
        if not od or od.get("type") is None:
            self.axes[0].text(0.5, 0.5, "Waiting...", transform=self.axes[0].transAxes,
                              ha="center", va="center", color=T.DIM)
            self.draw_idle()
            return
        t = od["type"]
        if t == "boundary":
            self._plot_boundary(od)
        elif t == "seq":
            self._plot_seq(od)
        elif t == "gan":
            self._plot_gan(od)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()

    def _plot_boundary(self, od):
        self._sty("Decision Boundary")
        gx, gy, gp = od["gx"], od["gy"], od["gp"]
        r = int(math.sqrt(len(gx)))
        try:
            self.axes[0].contourf(gx.reshape(r, r), gy.reshape(r, r),
                                   gp.reshape(r, r), levels=20, alpha=0.55, cmap="RdYlBu")
        except Exception:
            pass
        dx, dy, dc = od["dx"], od["dy"], od["dy2"]
        self.axes[0].scatter(dx[dc == 0], dy[dc == 0], c=T.PRI, s=5, edgecolors="w", linewidth=0.2, label="C0")
        self.axes[0].scatter(dx[dc == 1], dy[dc == 1], c=T.PEA, s=5, edgecolors="w", linewidth=0.2, label="C1")
        if len(np.unique(dc)) > 2:
            self.axes[0].scatter(dx[dc == 2], dy[dc == 2], c=T.GRN, s=5, edgecolors="w", linewidth=0.2, label="C2")
        self.axes[0].legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, markerscale=1.5)

    def _plot_seq(self, od):
        has_attn = "attention" in od and od["attention"] and od["attention"][0] is not None
        if has_attn:
            self.fig.clear()
            self.axes = [self.fig.add_subplot(121), self.fig.add_subplot(122)]
            for ax in self.axes:
                ax.set_facecolor(T.S1); ax.tick_params(colors=T.DIM, labelsize=6)
                for s in ax.spines.values():
                    s.set_color(T.OVR)
                ax.grid(True, alpha=0.1, color=T.DIM)
            self.axes[0].set_title("Sequence Prediction", color=T.SUB, fontsize=9, pad=5)
            act, pred = od["act"], od["pred"]
            cols = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, act.shape[0])):
                self.axes[0].plot(act[i, :, 0], color=cols[i], lw=1.1, label=f"T{i+1}")
                self.axes[0].plot(pred[i, :, 0], color=cols[i], lw=1.1, ls="--", alpha=0.5, label=f"P{i+1}")
            self.axes[0].legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, ncol=2)
            self.axes[1].set_title("Attention (Head 0, Layer 0)", color=T.SUB, fontsize=9, pad=5)
            self.axes[1].imshow(od["attention"][0], cmap="magma", aspect="auto", interpolation="nearest")
            self.axes[1].set_xlabel("Key", color=T.DIM, fontsize=6)
            self.axes[1].set_ylabel("Query", color=T.DIM, fontsize=6)
        else:
            self._sty("Sequence Prediction")
            act, pred = od["act"], od["pred"]
            cols = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, act.shape[0])):
                self.axes[0].plot(act[i, :, 0], color=cols[i], lw=1.1, label=f"True {i+1}")
                self.axes[0].plot(pred[i, :, 0], color=cols[i], lw=1.1, ls="--", alpha=0.5, label=f"Pred {i+1}")
            self.axes[0].legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, ncol=2)

    def _plot_gan(self, od):
        self._sty("GAN: Real vs Generated")
        r, g = od["real"], od["gen"]
        self.axes[0].scatter(r[:, 0], r[:, 1], c=T.PRI, s=4, alpha=0.5, label="Real", edgecolors="none")
        self.axes[0].scatter(g[:, 0], g[:, 1], c=T.RED, s=4, alpha=0.5, label="Gen", edgecolors="none")
        self.axes[0].set_xlim(-2.5, 2.5); self.axes[0].set_ylim(-2.5, 2.5)
        self.axes[0].set_aspect("equal")
        self.axes[0].legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)


class ConfPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Confusion Matrix")

    def update_plot(self, cd):
        self.clear_all()
        self._sty("Confusion Matrix")
        if not cd:
            self.draw_idle()
            return
        from factory import conf_matrix
        cm = conf_matrix(cd["true"], cd["pred"], cd["nc"])
        nc = cd["nc"]
        self.axes[0].imshow(cm, cmap="Blues", interpolation="nearest")
        self.axes[0].set_xticks(range(nc)); self.axes[0].set_yticks(range(nc))
        self.axes[0].set_xlabel("Pred", color=T.DIM, fontsize=6)
        self.axes[0].set_ylabel("True", color=T.DIM, fontsize=6)
        for i in range(nc):
            for j in range(nc):
                c = "white" if cm[i, j] > cm.max() / 2 else T.TXT
                self.axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color=c, fontsize=7, fontweight="bold")
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class FeatPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Feature Space (PCA)")

    def update_plot(self, fd):
        self.clear_all()
        self._sty("Feature Space (PCA)")
        if not fd:
            self.draw_idle()
            return
        x, y, lb = fd["x"], fd["y"], fd["labels"]
        cols = [T.PRI, T.PEA, T.GRN, T.MAU, T.YEL, T.TEAL]
        for c in np.unique(lb):
            m = lb == c
            self.axes[0].scatter(x[m], y[m], c=cols[int(c) % len(cols)], s=6, alpha=0.6,
                                 edgecolors="none", label=f"C{int(c)}")
        self.axes[0].legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class WeightPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Weight Distribution")

    def update_plot(self, wi):
        self.clear_all()
        self._sty("Weight Distribution")
        ms, ss = [], []
        for k, v in wi.items():
            if k.endswith("_m") and not k.endswith("_gm"):
                ms.append(v)
            if k.endswith("_s") and not k.endswith("_gs"):
                ss.append(v)
        if ms:
            x = range(len(ms))
            self.axes[0].bar(x, ms, color=T.MAU, alpha=0.7, label="Mean")
            self.axes[0].errorbar(x, ms, yerr=ss, color=T.PEA, fmt="none", capsize=3, elinewidth=0.7)
            self.axes[0].set_xticks(list(x))
            self.axes[0].set_xticklabels([str(i) for i in x], fontsize=5)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


# ═══════════════════════════════════════════════════════════════════════
#  Main Window
# ═══════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 NN Training Visualizer — PyTorch & TensorFlow")
        self.setMinimumSize(1320, 840)
        self.resize(1480, 900)
        self.setStyleSheet(T.css())
        self.ntype = "mlp"
        self.fw = "pytorch"
        self.thread = None
        self.trained = False
        self._build_ui()
        self._on_type_changed("mlp")
        self.statusBar().showMessage("Ready — select a network and press Start")

    # ── build UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        ml = QHBoxLayout(cw); ml.setContentsMargins(6, 6, 6, 6); ml.setSpacing(6)

        # Left panel (scrollable)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(278)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(4)

        # Framework
        g = QGroupBox("Deep Learning Framework"); v = QVBoxLayout(g)
        self.fw_combo = QComboBox()
        self.fw_combo.addItem("🔥 PyTorch"); self.fw_combo.setItemData(0, "pytorch")
        if TF_AVAILABLE:
            self.fw_combo.addItem("🧡 TensorFlow"); self.fw_combo.setItemData(1, "tensorflow")
        self.fw_combo.currentIndexChanged.connect(self._on_fw_changed)
        v.addWidget(self.fw_combo)
        self.device_lbl = QLabel(); self.device_lbl.setObjectName("desc")
        v.addWidget(self.device_lbl); ll.addWidget(g)

        # Architecture
        g = QGroupBox("Architecture"); v = QVBoxLayout(g)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["MLP", "CNN", "RNN", "LSTM", "Transformer", "GAN"])
        self.type_combo.currentTextChanged.connect(lambda t: self._on_type_changed(t.lower()))
        v.addWidget(self.type_combo)
        self.desc_lbl = QLabel(); self.desc_lbl.setObjectName("desc")
        self.desc_lbl.setWordWrap(True); self.desc_lbl.setMinimumHeight(44)
        v.addWidget(self.desc_lbl); ll.addWidget(g)

        # Optimizer
        g = QGroupBox("Optimizer"); v = QFormLayout(g)
        self.opt_combo = QComboBox(); self.opt_combo.addItems(["Adam", "SGD", "RMSprop", "AdamW"])
        v.addRow("Type:", self.opt_combo)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.01); self.lr_spin.setSingleStep(0.001); self.lr_spin.setDecimals(4)
        v.addRow("Learning Rate:", self.lr_spin)
        self.wd_spin = QDoubleSpinBox(); self.wd_spin.setRange(0, 0.1)
        self.wd_spin.setValue(0); self.wd_spin.setSingleStep(0.001); self.wd_spin.setDecimals(4)
        v.addRow("Weight Decay:", self.wd_spin); ll.addWidget(g)

        # Regularization
        g = QGroupBox("Regularization"); v = QVBoxLayout(g)
        self.dropout_cb = QCheckBox("Enable Dropout"); v.addWidget(self.dropout_cb)
        h = QHBoxLayout(); h.addWidget(QLabel("Rate:"))
        self.drop_spin = QDoubleSpinBox(); self.drop_spin.setRange(0, 0.8)
        self.drop_spin.setValue(0.2); self.drop_spin.setSingleStep(0.05)
        h.addWidget(self.drop_spin); v.addLayout(h); ll.addWidget(g)

        # Training
        g = QGroupBox("Training"); v = QFormLayout(g)
        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(10, 5000)
        self.epoch_spin.setValue(200); self.epoch_spin.setSingleStep(50)
        v.addRow("Epochs:", self.epoch_spin)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(4, 256)
        self.batch_spin.setValue(32); self.batch_spin.setSingleStep(8)
        v.addRow("Batch Size:", self.batch_spin)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 99999); self.seed_spin.setValue(42)
        v.addRow("Seed:", self.seed_spin)
        h = QHBoxLayout(); h.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(0, 100); self.speed_slider.setValue(5)
        h.addWidget(self.speed_slider); v.addLayout(h); ll.addWidget(g)

        # MLP layers
        g = QGroupBox("MLP Layers"); v = QVBoxLayout(g); self.mlp_group = g
        self.mlp_edit = QLineEdit("2, 64, 32, 2"); v.addWidget(self.mlp_edit); ll.addWidget(g)

        # Dataset
        g = QGroupBox("Dataset"); v = QVBoxLayout(g); self.data_group = g
        self.dataset_combo = QComboBox(); self.dataset_combo.addItems(["Circles", "Spirals", "XOR", "Moons"])
        v.addWidget(self.dataset_combo)
        self.data_lbl = QLabel(); self.data_lbl.setObjectName("desc"); self.data_lbl.setWordWrap(True)
        v.addWidget(self.data_lbl); ll.addWidget(g)

        # GAN target
        g = QGroupBox("GAN Target"); v = QVBoxLayout(g); self.gan_group = g
        self.gan_mode_combo = QComboBox(); self.gan_mode_combo.addItems(["Ring", "Spiral", "Gaussian Mix"])
        v.addWidget(self.gan_mode_combo); ll.addWidget(g)

        # Controls
        g = QGroupBox("Controls"); v = QVBoxLayout(g); v.setSpacing(4)
        r1 = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start"); self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self._start)
        self.pause_btn = QPushButton("⏸  Pause"); self.pause_btn.setObjectName("stopBtn")
        self.pause_btn.clicked.connect(self._pause); self.pause_btn.setEnabled(False)
        r1.addWidget(self.start_btn); r1.addWidget(self.pause_btn); v.addLayout(r1)
        r2 = QHBoxLayout()
        self.step_btn = QPushButton("→  Step"); self.step_btn.clicked.connect(self._step)
        self.reset_btn = QPushButton("↻  Reset"); self.reset_btn.setObjectName("resetBtn")
        self.reset_btn.clicked.connect(self._reset)
        r2.addWidget(self.step_btn); r2.addWidget(self.reset_btn); v.addLayout(r2)
        r3 = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save"); self.save_btn.clicked.connect(self._save_plots)
        self.export_btn = QPushButton("📋 CSV"); self.export_btn.clicked.connect(self._export_csv)
        r3.addWidget(self.save_btn); r3.addWidget(self.export_btn); v.addLayout(r3)
        ll.addWidget(g)

        # Info
        g = QGroupBox("Info"); v = QVBoxLayout(g)
        self.epoch_lbl = QLabel("Epoch: —"); self.epoch_lbl.setStyleSheet("font-size:12px;font-weight:bold;color:" + T.PRI)
        self.loss_lbl = QLabel("Loss: —"); self.loss_lbl.setStyleSheet("font-size:11px;color:" + T.PEA)
        self.metric_lbl = QLabel("Metric: —"); self.metric_lbl.setStyleSheet("font-size:11px;color:" + T.GRN)
        self.lr_lbl = QLabel("LR: —"); self.lr_lbl.setStyleSheet("font-size:11px;color:" + T.YEL)
        self.param_lbl = QLabel("Params: —"); self.param_lbl.setStyleSheet("font-size:10px;color:" + T.DIM)
        self.time_lbl = QLabel("Time: —"); self.time_lbl.setStyleSheet("font-size:10px;color:" + T.DIM)
        for w in [self.epoch_lbl, self.loss_lbl, self.metric_lbl, self.lr_lbl, self.param_lbl, self.time_lbl]:
            v.addWidget(w)
        ll.addWidget(g); ll.addStretch()
        scroll.setWidget(left); ml.addWidget(scroll)

        # Right panel
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
        self.arch_widget = ArchWidget(); rl.addWidget(self.arch_widget, stretch=2)

        self.tabs = QTabWidget()

        # Tab: Training
        t1 = QWidget(); g1 = QGridLayout(t1); g1.setContentsMargins(3, 3, 3, 3); g1.setSpacing(3)
        self.loss_plot = LossPlot(); self.metric_plot = MetricPlot()
        self.lr_plot = LRPlot(); self.grad_plot = GradFlowPlot()
        g1.addWidget(self.loss_plot, 0, 0); g1.addWidget(self.metric_plot, 0, 1)
        g1.addWidget(self.lr_plot, 1, 0); g1.addWidget(self.grad_plot, 1, 1)
        self.tabs.addTab(t1, "📈 Training")

        # Tab: Analysis
        t2 = QWidget(); g2 = QGridLayout(t2); g2.setContentsMargins(3, 3, 3, 3); g2.setSpacing(3)
        self.output_plot = OutputPlot(); self.conf_plot = ConfPlot()
        self.feat_plot = FeatPlot(); self.weight_plot = WeightPlot()
        g2.addWidget(self.output_plot, 0, 0, 1, 2)
        g2.addWidget(self.conf_plot, 1, 0); g2.addWidget(self.feat_plot, 1, 1)
        g2.addWidget(self.weight_plot, 2, 0, 1, 2)
        self.tabs.addTab(t2, "🔍 Analysis")

        # Tab: Log
        t3 = QWidget(); v3 = QVBoxLayout(t3); v3.setContentsMargins(3, 3, 3, 3)
        self.log_table = QTableWidget(); self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(["Epoch", "Loss", "Metric", "LR", "Avg‖∇‖"])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_table.setAlternatingRowColors(True)
        hh = self.log_table.horizontalHeader()
        for i in range(4):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        v3.addWidget(self.log_table); self.tabs.addTab(t3, "📊 Log")
        rl.addWidget(self.tabs, stretch=4); ml.addWidget(right, stretch=1)
        self._log_data = []
        self._update_device()

    # ── callbacks ───────────────────────────────────────────────────────
    def _update_device(self):
        if self.fw == "pytorch":
            import torch
            dev = "GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
            self.device_lbl.setText(f"Device: {dev}\nCUDA: {'Yes' if torch.cuda.is_available() else 'No'}")
        elif TF_AVAILABLE:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            self.device_lbl.setText(f"Device: {'GPU' if gpus else 'CPU'}\nGPUs: {len(gpus)}")
        else:
            self.device_lbl.setText("")

    def _on_fw_changed(self, idx):
        self.fw = self.fw_combo.itemData(idx) or "pytorch"
        if self.thread and self.thread.isRunning():
            self._reset()
        self._on_type_changed(self.ntype)
        self._update_device()

    def _on_type_changed(self, nt):
        if self.thread and self.thread.isRunning():
            self._reset()
        self.ntype = nt
        self.arch_widget.set_type(nt, self.fw, self._parse_mlp())
        self.desc_lbl.setText(DESCS.get(nt, ""))
        self.mlp_group.setVisible(nt == "mlp")
        self.data_group.setVisible(nt in ("mlp", "cnn"))
        self.gan_group.setVisible(nt == "gan")
        self.dropout_cb.setVisible(nt in ("mlp", "cnn"))
        self.data_lbl.setText(DATA_INFO.get(nt, ""))

    def _parse_mlp(self):
        try:
            ls = [int(x.strip()) for x in self.mlp_edit.text().split(",") if x.strip()]
            if len(ls) >= 2 and all(x > 0 for x in ls):
                return ls
        except Exception:
            pass
        return [2, 64, 32, 2]

    def _set_controls_enabled(self, enabled):
        self.start_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(False)
        self.step_btn.setEnabled(enabled)
        for w in [self.type_combo, self.dataset_combo, self.mlp_edit, self.gan_mode_combo, self.fw_combo]:
            w.setEnabled(enabled)

    def _reset_plots(self):
        for p in [self.loss_plot, self.metric_plot, self.lr_plot, self.grad_plot,
                   self.output_plot, self.conf_plot, self.feat_plot, self.weight_plot]:
            p.clear_all(); p.draw_idle()
        self.loss_plot.losses = []
        self.metric_plot.data = []
        self.lr_plot.data = []

    def _reset_info(self):
        self.pause_btn.setText("⏸  Pause")
        for l, t in [(self.epoch_lbl, "Epoch: —"), (self.loss_lbl, "Loss: —"),
                      (self.metric_lbl, "Metric: —"), (self.lr_lbl, "LR: —"),
                      (self.param_lbl, "Params: —"), (self.time_lbl, "Time: —")]:
            l.setText(t)

    # ── training actions ────────────────────────────────────────────────
    def _prepare_seed(self):
        seed = self.seed_spin.value()
        import torch
        np.random.seed(seed) if seed >= 0 else np.random.seed(None)
        torch.manual_seed(seed) if seed >= 0 else torch.seed()
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) if seed >= 0 else None
        if TF_AVAILABLE:
            import tensorflow as tf
            tf.random.set_seed(seed) if seed >= 0 else None

    def _build_kwargs(self, epochs):
        dr = self.drop_spin.value() if self.dropout_cb.isChecked() else 0.0
        model = make_model(self.ntype, self.fw, self._parse_mlp(), dr)
        if model is None:
            self.statusBar().showMessage("Model creation failed!")
            self._set_controls_enabled(True)
            return None
        kw = dict(ntype=self.ntype, fw=self.fw, model=model,
                  lr=self.lr_spin.value(), epochs=epochs,
                  bs=self.batch_spin.value(), wd=self.wd_spin.value())
        if self.ntype == "gan":
            mode = self.gan_mode_combo.currentText().lower().replace(" ", "_")
            kw["gd"] = make_gan_data(mode=mode)
            G, D = model
            self.param_lbl.setText(f"G:{count_params(G, self.fw)} D:{count_params(D, self.fw)}")
        elif self.ntype in ("mlp", "cnn"):
            ds = self.dataset_combo.currentText().lower()
            Xd, Yd = MLP_DATASETS[ds]() if self.ntype == "mlp" else make_signals()
            kw["Xd"] = Xd; kw["Yd"] = Yd
            nc = int(Yd.max()) + 1
            self.param_lbl.setText(f"Params: {count_params(model, self.fw)} | Classes: {nc}")
        else:
            Xd, Yd = make_sine()
            kw["Xd"] = Xd; kw["Yd"] = Yd
            self.param_lbl.setText(f"Params: {count_params(model, self.fw)}")
        return kw

    def _start(self):
        if self.trained:
            self._reset_plots()
        self._prepare_seed()
        self._set_controls_enabled(False)
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        kw = self._build_kwargs(self.epoch_spin.value())
        if kw is None:
            return
        self._t0 = QDateTime.currentDateTime()
        self.thread = TrainThread()
        self.thread.speed = self.speed_slider.value()
        self.thread.setup(**kw)
        self.thread.epoch_sig.connect(self._on_epoch)
        self.thread.finished_sig.connect(self._on_finished)
        self.thread.start()
        self.statusBar().showMessage(f"Training {self.ntype.upper()} [{self.fw}]...")

    def _step(self):
        if self.trained:
            self._reset_plots()
        self._prepare_seed()
        self._set_controls_enabled(False)
        self.start_btn.setEnabled(False)
        kw = self._build_kwargs(1)
        if kw is None:
            return
        self.thread = TrainThread()
        self.thread.speed = self.speed_slider.value()
        self.thread.setup(**kw)
        self.thread.epoch_sig.connect(self._on_epoch)
        self.thread.finished_sig.connect(self._on_step_done)
        self.thread.start()

    def _on_step_done(self):
        self._set_controls_enabled(True)
        self.trained = True
        self.statusBar().showMessage("Step done")

    def _pause(self):
        if not self.thread:
            return
        if self.pause_btn.text().startswith("⏸"):
            self.thread.pause(); self.pause_btn.setText("▶  Resume")
        else:
            self.thread.resume(); self.pause_btn.setText("⏸  Pause")

    def _reset(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop(); self.thread.wait(2000)
        self._reset_plots(); self._log_data = []; self.log_table.setRowCount(0)
        self.trained = False; self._set_controls_enabled(True); self._reset_info()
        self.statusBar().showMessage("Reset — Ready")

    def _on_finished(self):
        self._set_controls_enabled(True)
        self.trained = True
        self.pause_btn.setText("⏸  Pause")
        self.statusBar().showMessage(f"Training complete ✓ [{self.fw}]")

    # ── epoch callback ──────────────────────────────────────────────────
    def _on_epoch(self, ep, loss, metric, wi, od):
        lr = wi.get("lr", self.lr_spin.value())
        gn = wi.get("grad_norms", {})
        avg_gn = float(np.mean(list(gn.values()))) if gn else 0.0

        self.epoch_lbl.setText(f"Epoch: {ep}/{self.epoch_spin.value()}")
        self.loss_lbl.setText(f"Loss: {loss:.5f}")
        self.lr_lbl.setText(f"LR: {lr:.6f}")
        if self.ntype in ("mlp", "cnn"):
            self.metric_lbl.setText(f"Accuracy: {metric:.2%}")
        elif self.ntype == "gan":
            self.metric_lbl.setText(f"G Score: {metric:.4f}")
        else:
            self.metric_lbl.setText(f"R²: {metric:.4f}")
        dt = QDateTime.currentDateTime().msecsTo(self._t0)
        self.time_lbl.setText(f"Time: {abs(dt)/1000:.1f}s")

        self.loss_plot.update_plot(ep, loss)
        self.metric_plot.update_plot(ep, metric, self.ntype)
        self.lr_plot.update_plot(ep, lr)
        self.grad_plot.update_plot(gn)
        self.weight_plot.update_plot(wi)

        if od and od.get("type"):
            self.output_plot.update_plot(od)
            if "confusion" in od:
                self.conf_plot.update_plot(od["confusion"])
            if "features" in od:
                self.feat_plot.update_plot(od["features"])

        ms = f"{metric:.2%}" if self.ntype in ("mlp", "cnn") else f"{metric:.4f}"
        self._log_data.append([ep, f"{loss:.5f}", ms, f"{lr:.6f}", f"{avg_gn:.6f}"])
        if len(self._log_data) > 500:
            self._log_data = self._log_data[-500:]
        row = self.log_table.rowCount(); self.log_table.insertRow(row)
        for c, v in enumerate(self._log_data[-1]):
            it = QTableWidgetItem(str(v)); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.log_table.setItem(row, c, it)
        self.log_table.scrollToBottom()

        if self.thread:
            self.thread.speed = self.speed_slider.value()
        self.statusBar().showMessage(f"[{self.fw}] Epoch {ep} | Loss: {loss:.5f} | {ms}")

    # ── export ──────────────────────────────────────────────────────────
    def _save_plots(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "nn_training.png", "PNG (*.png)")
        if not path:
            return
        for name, p in [("loss", self.loss_plot), ("metric", self.metric_plot),
                         ("lr", self.lr_plot), ("grad", self.grad_plot),
                         ("output", self.output_plot), ("conf", self.conf_plot),
                         ("feat", self.feat_plot), ("weight", self.weight_plot)]:
            p.fig.savefig(path.replace(".png", f"_{name}.png"), dpi=150,
                          bbox_inches="tight", facecolor=T.S1)
        self.statusBar().showMessage(f"Saved to {path.replace('.png', '_*.png')}")

    def _export_csv(self):
        if not self._log_data:
            self.statusBar().showMessage("No data to export"); return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "log.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["Epoch", "Loss", "Metric", "LR", "Avg_Grad"])
            w.writerows(self._log_data)
        self.statusBar().showMessage(f"Exported to {path}")

    def closeEvent(self, e):
        if self.thread and self.thread.isRunning():
            self.thread.stop(); self.thread.wait(3000)
        e.accept()