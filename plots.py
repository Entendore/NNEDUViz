"""Matplotlib plot widgets for training visualization."""

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
        for spine in self.ax.spines.values():
            spine.set_color(T.OVR)
        self.ax.grid(True, alpha=0.1, color=T.DIM)
        if title:
            self.ax.set_title(title, color=T.SUB, fontsize=9, pad=5)

    def clear_all(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._apply_style()
        self.fig.tight_layout(pad=1.0)


class LossPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Training Loss")
        self.train_data = []
        self.test_data = []

    def clear_data(self):
        self.train_data.clear()
        self.test_data.clear()

    def update_plot(self, epoch: int, train_loss: float, test_loss: float = None):
        self.train_data.append((epoch, train_loss))
        if test_loss is not None:
            self.test_data.append((epoch, test_loss))
        self.clear_all()
        self._apply_style("Loss (Train / Test)")
        
        if self.train_data:
            epochs, losses = zip(*self.train_data)
            self.ax.plot(epochs, losses, color=T.PRI, lw=1.2, label="Train")
            self.ax.fill_between(epochs, losses, alpha=0.06, color=T.PRI)
        
        if self.test_data:
            epochs, losses = zip(*self.test_data)
            self.ax.plot(epochs, losses, color=T.RED, lw=1.2, ls="--", label="Test")
            self.ax.fill_between(epochs, losses, alpha=0.04, color=T.RED)
        
        if self.train_data or self.test_data:
            self.ax.legend(fontsize=6, facecolor=T.S1, edgecolor=T.OVR, 
                          labelcolor=T.TXT, loc="upper right")
            self.ax.set_xlabel("Epoch", color=T.DIM, fontsize=7)
            self.ax.set_ylabel("Loss", color=T.DIM, fontsize=7)
        
        self.draw_idle()


class MetricPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Metric")
        self.train_data = []
        self.test_data = []

    def clear_data(self):
        self.train_data.clear()
        self.test_data.clear()

    def update_plot(self, epoch: int, train_metric: float, test_metric: float = None, network_type: str = ""):
        self.train_data.append((epoch, train_metric))
        if test_metric is not None:
            self.test_data.append((epoch, test_metric))
        self.clear_all()
        
        labels = {
            "mlp": ("Accuracy", T.GRN), "cnn": ("Accuracy", T.GRN),
            "rnn": ("R-squared", T.PEA), "lstm": ("R-squared", T.PEA),
            "transformer": ("R-squared", T.PEA), "gan": ("G Score", T.RED),
        }
        label, color = labels.get(network_type, ("Metric", T.PEA))
        self._apply_style(f"{label} (Train / Test)")
        
        if self.train_data:
            epochs, values = zip(*self.train_data)
            self.ax.plot(epochs, values, color=color, lw=1.2, label="Train")
            self.ax.fill_between(epochs, values, alpha=0.06, color=color)
        
        if self.test_data:
            epochs, values = zip(*self.test_data)
            self.ax.plot(epochs, values, color=T.RED, lw=1.2, ls="--", label="Test")
            self.ax.fill_between(epochs, values, alpha=0.04, color=T.RED)
        
        if self.train_data or self.test_data:
            self.ax.legend(fontsize=6, facecolor=T.S1, edgecolor=T.OVR, 
                          labelcolor=T.TXT, loc="lower right")
            self.ax.set_xlabel("Epoch", color=T.DIM, fontsize=7)
            self.ax.set_ylabel(label, color=T.DIM, fontsize=7)
        
        self.draw_idle()


class LRPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Learning Rate")
        self.data = []

    def clear_data(self):
        self.data.clear()

    def update_plot(self, epoch: int, lr: float):
        self.data.append((epoch, lr))
        self.clear_all()
        self._apply_style("Learning Rate")
        if self.data:
            epochs, lrs = zip(*self.data)
            self.ax.plot(epochs, lrs, color=T.YEL, lw=1.2)
            self.ax.fill_between(epochs, lrs, alpha=0.08, color=T.YEL)
            self.ax.set_xlabel("Epoch", color=T.DIM, fontsize=7)
            self.ax.set_ylabel("LR", color=T.DIM, fontsize=7)
        self.draw_idle()


class GradFlowPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Gradient Flow")

    def update_plot(self, grad_norms: dict):
        self.clear_all()
        self._apply_style("Gradient Flow (log scale)")
        if not grad_norms:
            self.ax.text(0.5, 0.5, "No gradients yet...", transform=self.ax.transAxes,
                         ha="center", va="center", color=T.DIM)
            self.draw_idle()
            return
        names = list(grad_norms.keys())
        values = list(grad_norms.values())
        short_names = [n.split(".")[-1][:15] for n in names]
        colors = [T.GRN if v > 1e-4 else (T.YEL if v > 1e-6 else T.RED) for v in values]
        self.ax.barh(range(len(short_names)), values, color=colors, alpha=0.8)
        self.ax.set_yticks(range(len(short_names)))
        self.ax.set_yticklabels(short_names, fontsize=5)
        self.ax.set_xscale("log")
        self.ax.axvline(x=1e-6, color=T.RED, ls="--", alpha=0.4, lw=0.7)
        self.ax.text(1e-6, len(short_names) * 0.95, "vanishing", color=T.RED, fontsize=5, alpha=0.7)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class OutputPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Network Output")

    def update_plot(self, output_data: dict):
        self.clear_all()
        if not output_data or not output_data.get("type"):
            self.ax.text(0.5, 0.5, "Waiting for training...", transform=self.ax.transAxes,
                         ha="center", va="center", color=T.DIM)
            self.draw_idle()
            return
            
        plot_type = output_data["type"]
        
        if plot_type == "boundary":
            self._plot_boundary(output_data)
        elif plot_type == "seq":
            self._plot_sequence(output_data)
        elif plot_type == "gan":
            self._plot_gan(output_data)
            
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()

    def _plot_boundary(self, data: dict):
        self._apply_style("Decision Boundary (Train ● Test ○)")
        
        # FIX 1: Check if boundary data exists (only computed every N epochs)
        if "gx" not in data:
            self.ax.text(0.5, 0.5, "Computing boundary...", transform=self.ax.transAxes,
                         ha="center", va="center", color=T.DIM)
            return
            
        gx, gy, gp = data["gx"], data["gy"], data["gp"]
        r = int(math.sqrt(len(gx)))
        
        try:
            self.ax.contourf(gx.reshape(r, r), gy.reshape(r, r), gp.reshape(r, r),
                             levels=20, alpha=0.55, cmap="RdYlBu")
        except Exception:
            pass
        
        train_x = data.get("train_x", data["dx"])
        train_y = data.get("train_y", data["dy2"])
        test_x = data.get("test_x")
        test_y = data.get("test_y")
        class_colors = [T.PRI, T.PEA, T.GRN]
        
        for i, color in enumerate(class_colors):
            mask = train_y == i
            if mask.any():
                # FIX 2: Use train_y[mask] instead of train_y == i for Y coordinates
                self.ax.scatter(train_x[mask], train_y[mask], c=color, s=8, 
                               edgecolors="w", linewidths=0.3, marker="o",
                               label=f"Class {i} (train)")
        
        if test_x is not None and test_y is not None:
            for i, color in enumerate(class_colors):
                mask = test_y == i
                if mask.any():
                    # FIX 3: Use test_y[mask] instead of test_y == i for Y coordinates
                    self.ax.scatter(test_x[mask], test_y[mask], c=color, s=15, 
                                   edgecolors="w", linewidths=0.8, marker="o",
                                   facecolors="none", label=f"Class {i} (test)")
        
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, 
                      labelcolor=T.TXT, markerscale=1.2, loc="upper right")

    def _plot_sequence(self, data: dict):
        # FIX 4: Check if sequence data exists (only computed every N epochs)
        if "act" not in data:
            self.ax.text(0.5, 0.5, "Computing predictions...", transform=self.ax.transAxes,
                         ha="center", va="center", color=T.DIM)
            return
            
        has_attention = "attention" in data and data["attention"] and data["attention"][0] is not None
        test_pred = data.get("test_pred")
        test_act = data.get("test_act")
        
        # If we have test data, split into 3 panels
        if test_pred is not None and test_act is not None and has_attention:
            self.fig.clear()
            ax1 = self.fig.add_subplot(311)
            ax2 = self.fig.add_subplot(312)
            ax3 = self.fig.add_subplot(313)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_facecolor(T.S1)
                ax.tick_params(colors=T.DIM, labelsize=6)
                for spine in ax.spines.values():
                    spine.set_color(T.OVR)
                ax.grid(True, alpha=0.1, color=T.DIM)
                
            ax1.set_title("Train Predictions", color=T.SUB, fontsize=9, pad=5)
            actual, predicted = data["act"], data["pred"]
            colors = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, actual.shape[0])):
                ax1.plot(actual[i, :, 0], color=colors[i], lw=1.1)
                ax1.plot(predicted[i, :, 0], color=colors[i], lw=1.1, ls="--", alpha=0.5)
                
            ax2.set_title("Test Predictions", color=T.RED, fontsize=9, pad=5)
            for i in range(min(3, test_act.shape[0])):
                ax2.plot(test_act[i, :, 0], color=T.RED, lw=1.1)
                ax2.plot(test_pred[i, :, 0], color=T.RED, lw=1.1, ls="--", alpha=0.5)
                
            ax3.set_title("Self-Attention (Head 0, Layer 0)", color=T.SUB, fontsize=9, pad=5)
            ax3.imshow(data["attention"][0], cmap="magma", aspect="auto", interpolation="nearest")
            ax3.set_xlabel("Key position", color=T.DIM, fontsize=6)
            
        elif has_attention:
            self.fig.clear()
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            for ax in [ax1, ax2]:
                ax.set_facecolor(T.S1)
                ax.tick_params(colors=T.DIM, labelsize=6)
                for spine in ax.spines.values():
                    spine.set_color(T.OVR)
                ax.grid(True, alpha=0.1, color=T.DIM)
            ax1.set_title("Sequence Prediction (Train)", color=T.SUB, fontsize=9, pad=5)
            actual, predicted = data["act"], data["pred"]
            colors = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, actual.shape[0])):
                ax1.plot(actual[i, :, 0], color=colors[i], lw=1.1, label=f"True {i + 1}")
                ax1.plot(predicted[i, :, 0], color=colors[i], lw=1.1, ls="--", alpha=0.5, label=f"Pred {i + 1}")
            ax1.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, ncol=2)
            
            ax2.set_title("Self-Attention (Head 0, Layer 0)", color=T.SUB, fontsize=9, pad=5)
            ax2.imshow(data["attention"][0], cmap="magma", aspect="auto", interpolation="nearest")
            ax2.set_xlabel("Key position", color=T.DIM, fontsize=6)
            ax2.set_ylabel("Query position", color=T.DIM, fontsize=6)
        else:
            self._apply_style("Sequence Prediction (Train)")
            actual, predicted = data["act"], data["pred"]
            colors = [T.PRI, T.GRN, T.PEA]
            for i in range(min(3, actual.shape[0])):
                self.ax.plot(actual[i, :, 0], color=colors[i], lw=1.1, label=f"True {i + 1}")
                self.ax.plot(predicted[i, :, 0], color=colors[i], lw=1.1, ls="--", alpha=0.5, label=f"Pred {i + 1}")
            self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT, ncol=2)
            self.ax.set_xlabel("Time step", color=T.DIM, fontsize=7)

    def _plot_gan(self, data: dict):
        self._apply_style("GAN: Real vs Generated")
        
        # FIX 5: Check if generated data exists (only computed every N epochs)
        if "gen" not in data:
            self.ax.text(0.5, 0.5, "Generating samples...", transform=self.ax.transAxes,
                         ha="center", va="center", color=T.DIM)
            return
            
        real, generated = data["real"], data["gen"]
        self.ax.scatter(real[:, 0], real[:, 1], c=T.PRI, s=4, alpha=0.5, label="Real", edgecolors="none")
        self.ax.scatter(generated[:, 0], generated[:, 1], c=T.RED, s=4, alpha=0.5, label="Generated", edgecolors="none")
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_aspect("equal")
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)


class ConfusionMatrixPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Confusion Matrix (Test)")

    def update_plot(self, conf_data: dict):
        self.clear_all()
        title = "Confusion Matrix"
        if conf_data.get("is_test"):
            title += " (Test Set)"
        else:
            title += " (Train Set)"
        self._apply_style(title)
        if not conf_data:
            self.draw_idle()
            return
        cm = confusion_matrix(conf_data["true"], conf_data["pred"], conf_data["nc"])
        n_classes = conf_data["nc"]
        self.ax.imshow(cm, cmap="Blues", interpolation="nearest")
        self.ax.set_xticks(range(n_classes))
        self.ax.set_yticks(range(n_classes))
        self.ax.set_xlabel("Predicted", color=T.DIM, fontsize=6)
        self.ax.set_ylabel("True", color=T.DIM, fontsize=6)
        for i in range(n_classes):
            for j in range(n_classes):
                text_color = "white" if cm[i, j] > cm.max() / 2 else T.TXT
                self.ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color, fontsize=7, fontweight="bold")
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class FeatureSpacePlot(PlotCanvas):
    def __init__(self):
        super().__init__("Feature Space (PCA)")

    def update_plot(self, feat_data: dict):
        self.clear_all()
        is_test = feat_data.get("is_test", False)
        title = "Feature Space (PCA) - Test" if is_test else "Feature Space (PCA) - Train"
        self._apply_style(title)
        if not feat_data:
            self.draw_idle()
            return
        x, y, labels = feat_data["x"], feat_data["y"], feat_data["labels"]
        colors = [T.PRI, T.PEA, T.GRN, T.MAU, T.YEL, T.TEAL]
        for class_idx in np.unique(labels):
            mask = labels == class_idx
            self.ax.scatter(x[mask], y[mask], c=colors[int(class_idx) % len(colors)],
                            s=6, alpha=0.6, edgecolors="none", label=f"Class {int(class_idx)}")
        self.ax.legend(fontsize=5, facecolor=T.S1, edgecolor=T.OVR, labelcolor=T.TXT)
        self.ax.set_xlabel("PC1", color=T.DIM, fontsize=7)
        self.ax.set_ylabel("PC2", color=T.DIM, fontsize=7)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()


class WeightDistributionPlot(PlotCanvas):
    def __init__(self):
        super().__init__("Weight Distribution")

    def update_plot(self, weight_info: dict):
        self.clear_all()
        self._apply_style("Weight Distribution")
        means, stds = [], []
        for key, value in weight_info.items():
            if key.endswith("_m") and not key.endswith("_gm"):
                means.append(value)
            if key.endswith("_s") and not key.endswith("_gs"):
                stds.append(value)
        if means:
            x = range(len(means))
            self.ax.bar(x, means, color=T.MAU, alpha=0.7, label="Mean")
            self.ax.errorbar(x, means, yerr=stds, color=T.PEA, fmt="none", capsize=3, elinewidth=0.7)
            self.ax.set_xticks(list(x))
            self.ax.set_xticklabels([f"L{i}" for i in x], fontsize=5)
            self.ax.set_ylabel("Weight value", color=T.DIM, fontsize=7)
        self.fig.tight_layout(pad=1.0)
        self.draw_idle()