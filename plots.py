"""Matplotlib plot widgets for training visualization including eigen analysis."""

import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use("QtAgg")

from theme import ThemeColors
from factory import simple_pca, confusion_matrix

T = ThemeColors


class PlotCanvas(FigureCanvas):
    def __init__(self, title: str = ""):
        self.fig = Figure(figsize=(4, 3), dpi=85)
        self.fig.patch.set_facecolor(T.S1)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._apply_style(title)
        self.fig.tight_layout(pad=1.0)

    def _apply_style(self, title: str = ""):
        self.ax.set_facecolor(T.S1)
        self.ax.tick_params(colors=T.DIM, labelsize=6)
        for spine in self.ax.spines.values(): spine.set_color(T.OVR)
        self.ax.grid(True, alpha=0.1, color=T.DIM)
        if title: self.ax.set_title(title, color=T.SUB, fontsize=9, pad=5)

    def clear_all(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._apply_style()
        self.fig.tight_layout(pad=1.0)


class LossPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Training Loss")
        self.train_data, self.test_data = [], []

    def clear_data(self):
        self.train_data.clear(); self.test_data.clear()

    def update_plot(self, epoch, train_loss, test_loss=None):
        self.train_data.append((epoch, train_loss))
        if test_loss is not None: self.test_data.append((epoch, test_loss))
        self.clear_all(); self._apply_style("Loss (Train / Test)")
        if self.train_data:
            e, l = zip(*self.train_data); self.ax.plot(e, l, color=T.PRI, lw=1.2, label="Train")
        if self.test_data:
            e, l = zip(*self.test_data); self.ax.plot(e, l, color=T.RED, lw=1.2, ls="--", label="Test")
        if self.train_data or self.test_data:
            self.ax.legend(fontsize=6, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, loc="upper right")
        self.draw_idle()


class MetricPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Metric")
        self.train_data, self.test_data = [], []

    def clear_data(self):
        self.train_data.clear(); self.test_data.clear()

    def update_plot(self, epoch, train_m, test_m=None, nt=""):
        self.train_data.append((epoch, train_m))
        if test_m is not None: self.test_data.append((epoch, test_m))
        self.clear_all()
        labels = {"mlp": ("Accuracy", T.GRN), "cnn": ("Accuracy", T.GRN), "rnn": ("R²", T.PEA), "lstm": ("R²", T.PEA), "transformer": ("R²", T.PEA), "gan": ("G Score", T.RED)}
        label, color = labels.get(nt, ("Metric", T.PEA))
        self._apply_style(f"{label} (Train / Test)")
        if self.train_data:
            e, v = zip(*self.train_data); self.ax.plot(e, v, color=color, lw=1.2, label="Train")
        if self.test_data:
            e, v = zip(*self.test_data); self.ax.plot(e, v, color=T.RED, lw=1.2, ls="--", label="Test")
        if self.train_data or self.test_data:
            self.ax.legend(fontsize=6, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, loc="lower right")
        self.draw_idle()


class LRPlot(PlotCanvas):
    def __init__(self): super().__init__("Learning Rate"); self.data = []
    def clear_data(self): self.data.clear()
    def update_plot(self, epoch, lr):
        self.data.append((epoch, lr)); self.clear_all(); self._apply_style("Learning Rate")
        if self.data:
            e, l = zip(*self.data); self.ax.plot(e, l, color=T.YEL, lw=1.2)
        self.draw_idle()


class GradFlowPlot(PlotCanvas):
    def __init__(self): super().__init__("Gradient Flow")
    def update_plot(self, grad_norms):
        self.clear_all(); self._apply_style("Gradient Flow (log scale)")
        if not grad_norms:
            self.ax.text(0.5, 0.5, "No gradients yet...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM); self.draw_idle(); return
        names, values = list(grad_norms.keys()), list(grad_norms.values())
        short_names = [n.split(".")[-1][:15] for n in names]
        colors = [T.GRN if v > 1e-4 else (T.YEL if v > 1e-6 else T.RED) for v in values]
        self.ax.barh(range(len(short_names)), values, color=colors, alpha=0.8)
        self.ax.set_yticks(range(len(short_names))); self.ax.set_yticklabels(short_names, fontsize=5)
        self.ax.set_xscale("log"); self.ax.axvline(x=1e-6, color=T.RED, ls="--", alpha=0.4, lw=0.7)
        self.fig.tight_layout(pad=1.0); self.draw_idle()


class OutputPlot(PlotCanvas):
    def __init__(self): super().__init__("Network Output")

    def update_plot(self, output_data, eigen_arrows=None):
        self.clear_all()
        if not output_data or not output_data.get("type"):
            self.ax.text(0.5, 0.5, "Waiting for training...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM); self.draw_idle(); return
        pt = output_data["type"]
        if pt == "boundary": self._plot_boundary(output_data, eigen_arrows)
        elif pt == "seq": self._plot_sequence(output_data)
        elif pt == "gan": self._plot_gan(output_data)
        self.fig.tight_layout(pad=1.0); self.draw_idle()

    def _plot_boundary(self, data, eigen_arrows=None):
        self._apply_style("Decision Boundary (Train ● Test ○)")
        if "gx" not in data:
            self.ax.text(0.5, 0.5, "Computing boundary...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM); return
        gx, gy, gp = data["gx"], data["gy"], data["gp"]
        r = int(math.sqrt(len(gx)))
        try: self.ax.contourf(gx.reshape(r, r), gy.reshape(r, r), gp.reshape(r, r), levels=20, alpha=0.55, cmap="RdYlBu")
        except: pass
        train_x, train_y = data.get("train_x", data["dx"]), data.get("train_y", data["dy2"])
        test_x, test_y = data.get("test_x"), data.get("test_y")
        class_colors = [T.PRI, T.PEA, T.GRN]
        for i, color in enumerate(class_colors):
            mask = train_y == i
            if mask.any(): self.ax.scatter(train_x[mask], train_y[mask], c=color, s=8, edgecolors="w", linewidths=0.3, label=f"Class {i}")
        if test_x is not None and test_y is not None:
            for i, color in enumerate(class_colors):
                mask = test_y == i
                if mask.any(): self.ax.scatter(test_x[mask], test_y[mask], c=color, s=15, edgecolors="w", linewidths=0.8, marker="o", facecolors="none", label=f"Class {i} (test)")
        
        # Draw Eigenvector Arrows
        if eigen_arrows:
            arrow_colors = [T.MAU, T.YEL, T.TEAL]
            origin_x, origin_y = train_x.mean(), train_y.mean()
            for arrow_info in eigen_arrows:
                vecs = arrow_info['vectors']
                svs = arrow_info['singular_values']
                for j, (vec, sv) in enumerate(zip(vecs, svs)):
                    if j >= 2: break  # Only draw top 2
                    scale = 0.5 * sv / max(svs[0], 1e-5)
                    self.ax.quiver(origin_x, origin_y, vec[0]*scale, vec[1]*scale, 
                                   angles='xy', scale_units='xy', scale=1, 
                                   color=arrow_colors[j % len(arrow_colors)], width=0.01, 
                                   headwidth=3, headlength=4, zorder=5)
        
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, markerscale=1.2, loc="upper right")

    def _plot_sequence(self, data):
        if "act" not in data:
            self.ax.text(0.5, 0.5, "Computing predictions...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM); return
        has_attention = "attention" in data and data["attention"] and data["attention"][0] is not None
        test_pred, test_act = data.get("test_pred"), data.get("test_act")
        if test_pred is not None and test_act is not None and has_attention:
            self.fig.clear(); axes = [self.fig.add_subplot(3, 1, i+1) for i in range(3)]
            for ax in axes: ax.set_facecolor(T.S1); ax.tick_params(colors=T.DIM, labelsize=6); ax.grid(True, alpha=0.1, color=T.DIM)
            axes[0].set_title("Train Predictions", color=T.SUB, fontsize=9)
            colors = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, data["act"].shape[0])):
                axes[0].plot(data["act"][i, :, 0], color=colors[i], lw=1.1); axes[0].plot(data["pred"][i, :, 0], color=colors[i], lw=1.1, ls="--", alpha=0.5)
            axes[1].set_title("Test Predictions", color=T.RED, fontsize=9)
            for i in range(min(3, test_act.shape[0])):
                axes[1].plot(test_act[i, :, 0], color=T.RED, lw=1.1); axes[1].plot(test_pred[i, :, 0], color=T.RED, lw=1.1, ls="--", alpha=0.5)
            axes[2].set_title("Self-Attention", color=T.SUB, fontsize=9); axes[2].imshow(data["attention"][0], cmap="magma", aspect="auto")
        elif has_attention:
            self.fig.clear(); ax1, ax2 = self.fig.add_subplot(121), self.fig.add_subplot(122)
            for ax in [ax1, ax2]: ax.set_facecolor(T.S1); ax.tick_params(colors=T.DIM, labelsize=6); ax.grid(True, alpha=0.1, color=T.DIM)
            ax1.set_title("Sequence (Train)", color=T.SUB, fontsize=9)
            for i in range(min(3, data["act"].shape[0])):
                ax1.plot(data["act"][i, :, 0], color=T.PRI, lw=1.1); ax1.plot(data["pred"][i, :, 0], color=T.PRI, lw=1.1, ls="--", alpha=0.5)
            ax2.set_title("Self-Attention", color=T.SUB, fontsize=9); ax2.imshow(data["attention"][0], cmap="magma", aspect="auto")
        else:
            self._apply_style("Sequence Prediction (Train)")
            colors = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, data["act"].shape[0])):
                self.ax.plot(data["act"][i, :, 0], color=colors[i], lw=1.1); self.ax.plot(data["pred"][i, :, 0], color=colors[i], lw=1.1, ls="--", alpha=0.5)

    def _plot_gan(self, data):
        self._apply_style("GAN: Real vs Generated")
        if "gen" not in data:
            self.ax.text(0.5, 0.5, "Generating...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM); return
        self.ax.scatter(data["real"][:, 0], data["real"][:, 1], c=T.PRI, s=4, alpha=0.5, label="Real", edgecolors="none")
        self.ax.scatter(data["gen"][:, 0], data["gen"][:, 1], c=T.RED, s=4, alpha=0.5, label="Generated", edgecolors="none")
        self.ax.set_xlim(-2.5, 2.5); self.ax.set_ylim(-2.5, 2.5); self.ax.set_aspect("equal")
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)


class ConfusionMatrixPlot(PlotCanvas):
    def __init__(self): super().__init__("Confusion Matrix")
    def update_plot(self, conf_data):
        self.clear_all()
        self._apply_style("Confusion Matrix" + (" (Test)" if conf_data.get("is_test") else " (Train)"))
        if not conf_data: self.draw_idle(); return
        cm = confusion_matrix(conf_data["true"], conf_data["pred"], conf_data["nc"])
        self.ax.imshow(cm, cmap="Blues", interpolation="nearest")
        for i in range(conf_data["nc"]):
            for j in range(conf_data["nc"]):
                self.ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else T.TXT, fontsize=7, fontweight="bold")
        self.fig.tight_layout(pad=1.0); self.draw_idle()


class FeatureSpacePlot(PlotCanvas):
    def __init__(self): super().__init__("Feature Space")
    def update_plot(self, feat_data):
        self.clear_all()
        self._apply_style(("Feature Space (PCA) - Test" if feat_data.get("is_test") else "Feature Space (PCA) - Train"))
        if not feat_data: self.draw_idle(); return
        x, y, labels = feat_data["x"], feat_data["y"], feat_data["labels"]
        colors = [T.PRI, T.PEA, T.GRN, T.MAU, T.YEL, T.TEAL]
        for ci in np.unique(labels):
            mask = labels == ci
            self.ax.scatter(x[mask], y[mask], c=colors[int(ci) % len(colors)], s=6, alpha=0.6, label=f"Class {int(ci)}")
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)
        self.fig.tight_layout(pad=1.0); self.draw_idle()


class WeightDistributionPlot(PlotCanvas):
    def __init__(self): super().__init__("Weight Distribution")
    def update_plot(self, weight_info):
        self.clear_all(); self._apply_style("Weight Distribution")
        means, stds = [v for k, v in weight_info.items() if k.endswith("_m")], [v for k, v in weight_info.items() if k.endswith("_s")]
        if means:
            x = range(len(means))
            self.ax.bar(x, means, color=T.MAU, alpha=0.7, label="Mean"); self.ax.errorbar(x, means, yerr=stds, color=T.PEA, fmt="none", capsize=3)
        self.fig.tight_layout(pad=1.0); self.draw_idle()


# ============================================================
# NEW EIGEN PLOTS
# ============================================================

class EigenSpectrumPlot(PlotCanvas):
    def __init__(self): super().__init__("Singular Value Spectrum")

    def update_plot(self, svd_data):
        self.clear_all(); self._apply_style("Singular Value Spectrum (log scale)")
        if not svd_data:
            self.ax.text(0.5, 0.5, "Waiting for eigen analysis...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM)
            self.draw_idle(); return
        
        layer_colors = [T.PRI, T.GRN, T.PEA, T.MAU, T.YEL]
        for i, (name, s_vals) in enumerate(svd_data.items()):
            short_name = name.split(".")[-1][:12]
            color = layer_colors[i % len(layer_colors)]
            self.ax.plot(range(len(s_vals)), s_vals, 'o-', color=color, ms=3, lw=1.2, label=short_name)
        
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Index", color=T.DIM, fontsize=7)
        self.ax.set_ylabel("Singular Value", color=T.DIM, fontsize=7)
        self.ax.legend(fontsize=6, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, loc="upper right")
        self.fig.tight_layout(pad=1.0); self.draw_idle()


class ConditionNumberPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Condition Number & Rank")
        self.history = {}  # {layer_name: [(epoch, cond, rank), ...]}

    def clear_data(self):
        self.history.clear()

    def update_plot(self, epoch, cond_data, rank_data):
        for name in cond_data:
            if name not in self.history: self.history[name] = []
            self.history[name].append((epoch, cond_data[name], rank_data.get(name, 0)))
        
        self.clear_all()
        
        # Split into 2 subplots
        self.fig.clear()
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        for ax in [ax1, ax2]:
            ax.set_facecolor(T.S1)
            ax.tick_params(colors=T.DIM, labelsize=6)
            for spine in ax.spines.values(): spine.set_color(T.OVR)
            ax.grid(True, alpha=0.1, color=T.DIM)
        
        layer_colors = [T.PRI, T.GRN, T.PEA, T.MAU, T.YEL]
        
        ax1.set_title("Condition Number", color=T.SUB, fontsize=9)
        ax2.set_title("Effective Rank", color=T.SUB, fontsize=9)
        
        for i, (name, hist) in enumerate(self.history.items()):
            short_name = name.split(".")[-1][:12]
            color = layer_colors[i % len(layer_colors)]
            epochs, conds, ranks = zip(*hist)
            ax1.plot(epochs, conds, color=color, lw=1.2, label=short_name)
            ax2.plot(epochs, ranks, color=color, lw=1.2, label=short_name)
            
        ax1.set_yscale("log")
        ax1.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)
        ax2.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)
        
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class LossLandscapePlot(PlotCanvas):
    def __init__(self): super().__init__("Loss Landscape Slice")

    def update_plot(self, landscape_data):
        self.clear_all(); self._apply_style("Loss Landscape Slice (1D)")
        if not landscape_data:
            self.ax.text(0.5, 0.5, "Computing landscape (expensive)...", transform=self.ax.transAxes, ha="center", va="center", color=T.DIM)
            self.draw_idle(); return
        
        for name, data in landscape_data.items():
            offsets, losses = data["offsets"], data["losses"]
            valid = ~np.isnan(losses)
            if np.any(valid):
                self.ax.plot(offsets[valid], losses[valid], color=T.PRI, lw=1.5)
                self.ax.fill_between(offsets[valid], losses[valid], losses[valid].min(), alpha=0.15, color=T.PRI)
                self.ax.axvline(x=0, color=T.YEL, ls="--", alpha=0.5, lw=0.7)
                self.ax.text(0.02, 0.95, f"Min: {losses[valid].min():.4f}\nMax: {losses[valid].max():.4f}", 
                             transform=self.ax.transAxes, color=T.GRN, fontsize=7, va="top")
        
        self.ax.set_xlabel("Perturbation α", color=T.DIM, fontsize=7)
        self.ax.set_ylabel("Loss", color=T.DIM, fontsize=7)
        self.fig.tight_layout(pad=1.0); self.draw_idle()