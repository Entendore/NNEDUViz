"""Background training thread — framework-agnostic dispatch."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtCore import QThread, Signal

from factory import simple_pca, conf_matrix, grid_2d
from models_tensorflow import TF_AVAILABLE

if TF_AVAILABLE:
    import tensorflow as tf


class TrainThread(QThread):
    epoch_sig = Signal(int, float, float, dict, object)
    finished_sig = Signal()

    def __init__(self):
        super().__init__()
        self._run = False
        self._pause = False
        self.speed = 5

    def setup(self, **kw):
        self.__dict__.update(kw)

    def run(self):
        self._run = True
        self._pause = False
        getattr(self, f"_train_{self.ntype}")()
        self.finished_sig.emit()

    # ── flow control ────────────────────────────────────────────────────
    def _wait(self):
        while self._pause:
            if not self._run:
                return False
            self.msleep(50)
        return self._run

    def _freq(self, ep):
        if ep <= 5: return 1
        if ep <= 20: return 2
        if ep <= 100: return 5
        return 10

    def stop(self):
        self._run = False

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    # ── classification dispatcher ───────────────────────────────────────
    def _train_mlp(self):
        self._train_cls()

    def _train_cnn(self):
        self._train_cls()

    def _train_cls(self):
        (self._train_cls_pt if self.fw == "pytorch" else self._train_cls_tf)()

    def _train_cls_pt(self):
        m = self.model
        m.train()
        X = torch.FloatTensor(self.Xd)
        y = torch.LongTensor(self.Yd)
        nc = int(y.max().item()) + 1
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.wd)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), self.bs, True)

        for ep in range(self.epochs):
            if not self._wait():
                break
            m.train()
            tl = 0
            for bx, by in dl:
                opt.zero_grad()
                l = crit(m(bx), by)
                l.backward()
                opt.step()
                tl += l.item()
            al = tl / len(dl)
            gn = {n: float(p.grad.norm()) for n, p in m.named_parameters() if p.grad is not None}
            wi = {"lr": self.lr, "grad_norms": gn}
            wi.update(_wi_pt(m))
            with torch.no_grad():
                m.eval()
                preds = m(X).argmax(1)
                acc = float((preds == y).float().mean())
                m.train()
            od = self._cls_od_pt(m, X, y, nc, ep)
            self.epoch_sig.emit(ep + 1, al, acc, wi, od)
            self.msleep(self.speed)

    def _train_cls_tf(self):
        m = self.model
        Xd, Yd = self.Xd, self.Yd
        nc = int(Yd.max()) + 1
        m.build((None,) + Xd.shape[1:])
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        crit = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ds = tf.data.Dataset.from_tensor_slices((Xd, Yd)).shuffle(len(Xd)).batch(self.bs)

        for ep in range(self.epochs):
            if not self._wait():
                break
            tl = 0
            for bx, by in ds:
                with tf.GradientTape() as tape:
                    l = crit(by, m(bx, training=True))
                grads = tape.gradient(l, m.trainable_variables)
                opt.apply_gradients(zip(grads, m.trainable_variables))
                tl += float(l)
            al = tl / max(1, len(Xd) // self.bs)
            gn = {f"l{i}": float(tf.norm(g).numpy()) for i, g in enumerate(grads) if g is not None}
            wi = {"lr": float(opt.learning_rate.numpy()), "grad_norms": gn}
            wi.update(_wi_tf(m))
            preds = np.argmax(m(Xd, training=False).numpy(), axis=1)
            acc = float((preds == Yd).mean())
            od = self._cls_od_tf(m, Xd, Yd, nc, ep)
            self.epoch_sig.emit(ep + 1, al, acc, wi, od)
            self.msleep(self.speed)

    def _cls_od_pt(self, m, X, y, nc, ep):
        f = self._freq(ep)
        od = {"type": "boundary"}
        if ep % f == 0 or ep == self.epochs - 1:
            with torch.no_grad():
                gx, gy, gpts = grid_2d(X[:, :2].numpy())
                g = torch.FloatTensor(gpts)
                gp = m(g).argmax(1).numpy()
                od.update(gx=gx.ravel(), gy=gy.ravel(), gp=gp,
                          dx=X[:, 0].numpy(), dy=X[:, 1].numpy(), dy2=y.numpy())
                p = m(X).argmax(1)
                od["confusion"] = {"true": y.numpy(), "pred": p.numpy(), "nc": nc}
                try:
                    feat = m.get_features(X)
                    fp = simple_pca(feat)
                    od["features"] = {"x": fp[:, 0], "y": fp[:, 1], "labels": y.numpy()}
                except Exception:
                    pass
        return od

    def _cls_od_tf(self, m, Xd, Yd, nc, ep):
        f = self._freq(ep)
        od = {"type": "boundary"}
        if ep % f == 0 or ep == self.epochs - 1:
            gx, gy, gpts = grid_2d(Xd[:, :2])
            gp = np.argmax(m(gpts, training=False).numpy(), axis=1)
            od.update(gx=gx.ravel(), gy=gy.ravel(), gp=gp,
                      dx=Xd[:, 0], dy=Xd[:, 1], dy2=Yd)
            preds = np.argmax(m(Xd, training=False).numpy(), axis=1)
            od["confusion"] = {"true": Yd, "pred": preds, "nc": nc}
            try:
                feat = m.get_features(Xd.astype(np.float32))
                fp = simple_pca(feat)
                od["features"] = {"x": fp[:, 0], "y": fp[:, 1], "labels": Yd}
            except Exception:
                pass
        return od

    # ── sequence dispatcher ─────────────────────────────────────────────
    def _train_rnn(self):
        self._train_seq()

    def _train_lstm(self):
        self._train_seq()

    def _train_transformer(self):
        self._train_seq()

    def _train_seq(self):
        (self._train_seq_pt if self.fw == "pytorch" else self._train_seq_tf)()

    def _train_seq_pt(self):
        m = self.model
        m.train()
        X = torch.FloatTensor(self.Xd)
        y = torch.FloatTensor(self.Yd)
        crit = nn.MSELoss()
        opt = optim.Adam(m.parameters(), lr=self.lr, weight_decay=self.wd)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), self.bs, True)

        for ep in range(self.epochs):
            if not self._wait():
                break
            m.train()
            tl = 0
            for bx, by in dl:
                opt.zero_grad()
                l = crit(m(bx), by)
                l.backward()
                opt.step()
                tl += l.item()
            al = tl / len(dl)
            gn = {n: float(p.grad.norm()) for n, p in m.named_parameters() if p.grad is not None}
            wi = {"lr": self.lr, "grad_norms": gn}
            wi.update(_wi_pt(m))
            f = self._freq(ep)
            od = {"type": "seq"}
            with torch.no_grad():
                m.eval()
                p = m(X)
                met = 1 - float(torch.mean((p - y) ** 2))
                m.train()
                if ep % f == 0 or ep == self.epochs - 1:
                    od["act"] = y[:3].numpy()
                    od["pred"] = p[:3].numpy()
                if self.ntype == "transformer" and hasattr(m, "all_attn") and m.all_attn:
                    if ep % f == 0:
                        od["attention"] = [w[0].numpy() for w in m.all_attn if w is not None]
            self.epoch_sig.emit(ep + 1, al, met, wi, od)
            self.msleep(self.speed)

    def _train_seq_tf(self):
        m = self.model
        Xd, Yd = self.Xd, self.Yd
        m.build((None,) + Xd.shape[1:])
        opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        crit = tf.keras.losses.MeanSquaredError()
        ds = tf.data.Dataset.from_tensor_slices((Xd, Yd)).shuffle(len(Xd)).batch(self.bs)

        for ep in range(self.epochs):
            if not self._wait():
                break
            tl = 0
            for bx, by in ds:
                with tf.GradientTape() as tape:
                    l = crit(by, m(bx, training=True))
                grads = tape.gradient(l, m.trainable_variables)
                opt.apply_gradients(zip(grads, m.trainable_variables))
                tl += float(l)
            al = tl / max(1, len(Xd) // self.bs)
            gn = {f"l{i}": float(tf.norm(g).numpy()) for i, g in enumerate(grads) if g is not None}
            wi = {"lr": float(opt.learning_rate.numpy()), "grad_norms": gn}
            wi.update(_wi_tf(m))
            p = m(Xd, training=False).numpy()
            met = 1 - float(np.mean((p - Yd) ** 2))
            f = self._freq(ep)
            od = {"type": "seq"}
            if ep % f == 0 or ep == self.epochs - 1:
                od["act"] = Yd[:3]
                od["pred"] = p[:3]
            if self.ntype == "transformer" and hasattr(m, "get_attention"):
                if ep % f == 0:
                    _ = m(Xd[:1], training=False)
                    attn = m.get_attention()
                    od["attention"] = [a[0] for a in attn if a is not None]
            self.epoch_sig.emit(ep + 1, al, met, wi, od)
            self.msleep(self.speed)

    # ── GAN ─────────────────────────────────────────────────────────────
    def _train_gan(self):
        (self._train_gan_pt if self.fw == "pytorch" else self._train_gan_tf)()

    def _train_gan_pt(self):
        G, D = self.model
        rd = torch.FloatTensor(self.gd)
        n, ld = len(rd), 16
        oG = optim.Adam(G.parameters(), lr=self.lr)
        oD = optim.Adam(D.parameters(), lr=self.lr)
        crit = nn.BCELoss()

        for ep in range(self.epochs):
            if not self._wait():
                break
            # Discriminator step
            D.train()
            oD.zero_grad()
            idx = np.random.choice(n, self.bs, replace=False)
            real = rd[idx]
            fake = G(torch.randn(self.bs, ld)).detach()
            dloss = crit(D(real), torch.ones(self.bs, 1)) + crit(D(fake), torch.zeros(self.bs, 1))
            dloss.backward()
            oD.step()
            # Generator step
            G.train()
            oG.zero_grad()
            fake = G(torch.randn(self.bs, ld))
            gloss = crit(D(fake), torch.ones(self.bs, 1))
            gloss.backward()
            oG.step()
            # Gradient norms
            gn = {}
            for n2, p2 in G.named_parameters():
                if p2.grad is not None:
                    gn["G_" + n2] = float(p2.grad.norm())
            for n2, p2 in D.named_parameters():
                if p2.grad is not None:
                    gn["D_" + n2] = float(p2.grad.norm())
            wi = {"lr": self.lr, "grad_norms": gn}
            wi.update(_wi_pt(G, "G_"))
            wi.update(_wi_pt(D, "D_"))
            od = {"type": "gan", "real": rd.numpy()}
            f = self._freq(ep)
            if ep % f == 0 or ep == self.epochs - 1:
                with torch.no_grad():
                    G.eval()
                    od["gen"] = G(torch.randn(200, ld)).numpy()
                    G.train()
            self.epoch_sig.emit(ep + 1, dloss.item() + gloss.item(), -gloss.item(), wi, od)
            self.msleep(self.speed)

    def _train_gan_tf(self):
        G, D = self.model
        gd = self.gd
        n, ld = len(gd), 16
        oG = tf.keras.optimizers.Adam(learning_rate=self.lr)
        oD = tf.keras.optimizers.Adam(learning_rate=self.lr)
        bc = tf.keras.losses.BinaryCrossentropy()
        G.build((None, ld))
        D.build((None, 2))

        for ep in range(self.epochs):
            if not self._wait():
                break
            # Discriminator step
            idx = np.random.choice(n, self.bs, replace=False)
            real = gd[idx]
            noise = np.random.randn(self.bs, ld).astype(np.float32)
            with tf.GradientTape() as tD:
                dl_r = bc(tf.ones((self.bs, 1)), D(real, training=True))
                fake = G(noise, training=True)
                dl_f = bc(tf.zeros((self.bs, 1)), D(fake, training=True))
                dloss = dl_r + dl_f
            dgrads = tD.gradient(dloss, D.trainable_variables)
            oD.apply_gradients(zip(dgrads, D.trainable_variables))
            # Generator step
            noise = np.random.randn(self.bs, ld).astype(np.float32)
            with tf.GradientTape() as tG:
                fake = G(noise, training=True)
                gloss = bc(tf.ones((self.bs, 1)), D(fake, training=True))
            ggrads = tG.gradient(gloss, G.trainable_variables)
            oG.apply_gradients(zip(ggrads, G.trainable_variables))
            # Gradient norms
            gn = {}
            for i, g in enumerate(dgrads):
                if g is not None:
                    gn[f"D_l{i}"] = float(tf.norm(g).numpy())
            for i, g in enumerate(ggrads):
                if g is not None:
                    gn[f"G_l{i}"] = float(tf.norm(g).numpy())
            wi = {"lr": float(oG.learning_rate.numpy()), "grad_norms": gn}
            wi.update(_wi_tf(G, "G_"))
            wi.update(_wi_tf(D, "D_"))
            od = {"type": "gan", "real": gd}
            f = self._freq(ep)
            if ep % f == 0 or ep == self.epochs - 1:
                od["gen"] = G(np.random.randn(200, ld).astype(np.float32), training=False).numpy()
            self.epoch_sig.emit(ep + 1, float(dloss + gloss), -float(gloss), wi, od)
            self.msleep(self.speed)


# ── weight-info helpers (module-private) ───────────────────────────────

def _wi_pt(m, prefix=""):
    info = {}
    for n, p in m.named_parameters():
        if p.data.numel() > 0:
            info[prefix + n + "_m"] = float(p.data.mean())
            info[prefix + n + "_s"] = float(p.data.std())
    return info


def _wi_tf(m, prefix=""):
    info = {}
    for i, v in enumerate(m.trainable_variables):
        a = v.numpy()
        info[prefix + f"l{i}_m"] = float(np.mean(a))
        info[prefix + f"l{i}_s"] = float(np.std(a))
    return info