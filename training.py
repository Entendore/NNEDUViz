"""Background training thread for PyTorch models with eigen analysis."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtCore import QThread, Signal
from factory import simple_pca, confusion_matrix, grid_2d
from eigen import compute_weight_svd, compute_eigen_arrows, compute_loss_landscape_slice


class TrainThread(QThread):
    epoch_sig = Signal(int, float, float, float, float, dict, object)
    finished_sig = Signal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._paused = False
        self.speed = 5

    def setup(self, **kwargs):
        self.__dict__.update(kwargs)

    def run(self):
        self._running = True
        self._paused = False
        trainer_map = {
            "mlp": self._train_classification, "cnn": self._train_classification,
            "rnn": self._train_sequence, "lstm": self._train_sequence, "transformer": self._train_sequence,
            "gan": self._train_gan,
        }
        trainer = trainer_map.get(self.ntype)
        if trainer:
            trainer()
        self.finished_sig.emit()

    def _wait_if_paused(self) -> bool:
        while self._paused:
            if not self._running: return False
            self.msleep(50)
        return self._running

    def _update_frequency(self, epoch: int) -> int:
        if epoch <= 5: return 1
        if epoch <= 20: return 2
        if epoch <= 100: return 5
        return 10

    def _eigen_frequency(self, epoch: int) -> int:
        if epoch <= 5: return 1
        if epoch <= 20: return 2
        return 5

    def _landscape_frequency(self, epoch: int) -> int:
        if epoch <= 10: return 5
        return 20

    def stop(self): self._running = False
    def pause(self): self._paused = True
    def resume(self): self._paused = False

    def _compute_eigen_data(self, model, epoch) -> dict:
        svd_results = compute_weight_svd(model)
        eigen_data = {"svd": {}, "condition_numbers": {}, "effective_ranks": {}, "top_eigenvectors": {}, "loss_landscape": None, "arrows": []}
        
        for name, data in svd_results.items():
            eigen_data["svd"][name] = data["singular_values"]
            eigen_data["condition_numbers"][name] = data["condition_number"]
            eigen_data["effective_ranks"][name] = data["effective_rank"]
            eigen_data["top_eigenvectors"][name] = data["Vt"]
        
        if hasattr(model, 'get_weight_matrices'):
            eigen_data["arrows"] = compute_eigen_arrows(svd_results, input_dim=2)
            
        return eigen_data

    def _compute_landscape(self, model, criterion, X, y, epoch) -> dict:
        svd_results = compute_weight_svd(model)
        landscape = {}
        
        first_layer, first_direction = None, None
        for name, data in svd_results.items():
            first_layer = name
            first_direction = data["Vt"][0]
            break
            
        if first_layer and first_direction is not None:
            try:
                offsets, losses = compute_loss_landscape_slice(model, criterion, X, y, first_layer, first_direction, n_points=31, scale=2.0)
                landscape[first_layer] = {"offsets": offsets, "losses": losses}
            except Exception:
                landscape = {}
        return landscape

    def _train_classification(self):
        model = self.model
        model.train()
        X_train = torch.FloatTensor(self.X_train)
        y_train = torch.LongTensor(self.Y_train)
        X_test = torch.FloatTensor(self.X_test) if self.X_test is not None else None
        y_test = torch.LongTensor(self.Y_test) if self.Y_test is not None else None
        n_classes = int(y_train.max().item()) + 1

        criterion = nn.CrossEntropyLoss()
        optimizer = self._get_optimizer(model)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), self.bs, shuffle=True)

        for epoch in range(self.epochs):
            if not self._wait_if_paused(): break
            model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(dataloader)

            grad_norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
            weight_info = {"lr": self._current_lr(optimizer), "grad_norms": grad_norms}
            weight_info.update(self._get_weight_stats(model))

            with torch.no_grad():
                model.eval()
                train_acc = float((model(X_train).argmax(1) == y_train).float().mean())
                test_loss, test_acc = None, None
                if X_test is not None:
                    test_loss = float(criterion(model(X_test), y_test))
                    test_acc = float((model(X_test).argmax(1) == y_test).float().mean())
                model.train()

            output_data = self._get_classification_output(model, X_train, y_train, X_test, y_test, n_classes, epoch)

            eigen_freq = self._eigen_frequency(epoch)
            if epoch % eigen_freq == 0 or epoch == self.epochs - 1:
                eigen_data = self._compute_eigen_data(model, epoch)
                landscape_freq = self._landscape_frequency(epoch)
                if epoch % landscape_freq == 0 or epoch == self.epochs - 1:
                    try:
                        eigen_data["loss_landscape"] = self._compute_landscape(model, criterion, X_train[:64], y_train[:64], epoch)
                    except Exception: pass
                weight_info["eigen"] = eigen_data

            self.epoch_sig.emit(epoch + 1, train_loss, train_acc, test_loss, test_acc, weight_info, output_data)
            self.msleep(self.speed)

    def _get_classification_output(self, model, X_train, y_train, X_test, y_test, n_classes, epoch):
        freq = self._update_frequency(epoch)
        output = {"type": "boundary"}
        if epoch % freq == 0 or epoch == self.epochs - 1:
            with torch.no_grad():
                model.eval()
                gx, gy, grid_points = grid_2d(X_train[:, :2].numpy())
                output.update(gx=gx.ravel(), gy=gy.ravel(), gp=model(torch.FloatTensor(grid_points)).argmax(1).numpy(),
                              train_x=X_train[:, 0].numpy(), train_y=y_train.numpy(), dx=X_train[:, 0].numpy(), dy=X_train[:, 1].numpy(), dy2=y_train.numpy())
                if X_test is not None:
                    output["test_x"], output["test_y"] = X_test[:, 0].numpy(), y_test.numpy()
                    output["confusion"] = {"true": y_test.numpy(), "pred": model(X_test).argmax(1).numpy(), "nc": n_classes, "is_test": True}
                    try:
                        features = model.get_features(X_test)
                        pca_features = simple_pca(features)
                        output["features"] = {"x": pca_features[:, 0], "y": pca_features[:, 1], "labels": y_test.numpy(), "is_test": True}
                    except Exception: pass
                else:
                    output["confusion"] = {"true": y_train.numpy(), "pred": model(X_train).argmax(1).numpy(), "nc": n_classes, "is_test": False}
                model.train()
        return output

    def _train_sequence(self):
        model = self.model; model.train()
        X_train, y_train = torch.FloatTensor(self.X_train), torch.FloatTensor(self.Y_train)
        X_test = torch.FloatTensor(self.X_test) if self.X_test is not None else None
        y_test = torch.FloatTensor(self.Y_test) if self.Y_test is not None else None

        criterion = nn.MSELoss()
        optimizer = self._get_optimizer(model)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), self.bs, shuffle=True)

        for epoch in range(self.epochs):
            if not self._wait_if_paused(): break
            model.train(); total_loss = 0
            for bx, by in dataloader:
                optimizer.zero_grad(); loss = criterion(model(bx), by); loss.backward(); optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(dataloader)

            grad_norms = {n: float(p.grad.norm()) for n, p in model.named_parameters() if p.grad is not None}
            weight_info = {"lr": self._current_lr(optimizer), "grad_norms": grad_norms}
            weight_info.update(self._get_weight_stats(model))

            freq = self._update_frequency(epoch)
            output = {"type": "seq"}
            with torch.no_grad():
                model.eval()
                preds = model(X_train)
                ss_res, ss_tot = torch.sum((preds - y_train)**2), torch.sum((y_train - y_train.mean())**2)
                train_r2 = 1 - float(ss_res / ss_tot) if ss_tot > 0 else 0.0
                test_loss, test_r2 = None, None
                if X_test is not None:
                    test_loss = float(criterion(model(X_test), y_test))
                    tp = model(X_test); ss_t = torch.sum((tp - y_test)**2); ss_to = torch.sum((y_test - y_test.mean())**2)
                    test_r2 = 1 - float(ss_t / ss_to) if ss_to > 0 else 0.0
                model.train()

                if epoch % freq == 0 or epoch == self.epochs - 1:
                    output["act"], output["pred"] = y_train[:3].numpy(), preds[:3].numpy()
                    if X_test is not None:
                        output["test_act"], output["test_pred"] = y_test[:3].numpy(), model(X_test)[:3].numpy()
                if self.ntype == "transformer" and hasattr(model, "all_attn") and model.all_attn and epoch % freq == 0:
                    output["attention"] = [w[0].numpy() for w in model.all_attn if w is not None]

            eigen_freq = self._eigen_frequency(epoch)
            if epoch % eigen_freq == 0 or epoch == self.epochs - 1:
                eigen_data = self._compute_eigen_data(model, epoch)
                landscape_freq = self._landscape_frequency(epoch)
                if epoch % landscape_freq == 0 or epoch == self.epochs - 1:
                    try: eigen_data["loss_landscape"] = self._compute_landscape(model, criterion, X_train[:32], y_train[:32], epoch)
                    except Exception: pass
                weight_info["eigen"] = eigen_data

            self.epoch_sig.emit(epoch + 1, train_loss, train_r2, test_loss, test_r2, weight_info, output)
            self.msleep(self.speed)

    def _train_gan(self):
        Generator, Discriminator = self.model
        real_data = torch.FloatTensor(self.gd)
        n_samples, latent_dim = len(real_data), 16
        opt_G, opt_D = optim.Adam(Generator.parameters(), lr=self.lr), optim.Adam(Discriminator.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            if not self._wait_if_paused(): break
            
            Discriminator.train(); opt_D.zero_grad()
            idx = np.random.choice(n_samples, self.bs, replace=False)
            real_batch = real_data[idx]
            d_loss_real = criterion(Discriminator(real_batch), torch.ones(self.bs, 1))
            fake_samples = Generator(torch.randn(self.bs, latent_dim)).detach()
            d_loss_fake = criterion(Discriminator(fake_samples), torch.zeros(self.bs, 1))
            (d_loss_real + d_loss_fake).backward(); opt_D.step()

            Generator.train(); opt_G.zero_grad()
            fake_samples = Generator(torch.randn(self.bs, latent_dim))
            g_loss = criterion(Discriminator(fake_samples), torch.ones(self.bs, 1))
            g_loss.backward(); opt_G.step()

            grad_norms = {}
            for n, p in Generator.named_parameters():
                if p.grad is not None: grad_norms["G_" + n] = float(p.grad.norm())
            for n, p in Discriminator.named_parameters():
                if p.grad is not None: grad_norms["D_" + n] = float(p.grad.norm())
            weight_info = {"lr": self.lr, "grad_norms": grad_norms}
            weight_info.update(self._get_weight_stats(Generator, "G_"))
            weight_info.update(self._get_weight_stats(Discriminator, "D_"))

            freq = self._update_frequency(epoch)
            output = {"type": "gan", "real": real_data.numpy()}
            if epoch % freq == 0 or epoch == self.epochs - 1:
                with torch.no_grad():
                    Generator.eval()
                    output["gen"] = Generator(torch.randn(200, latent_dim)).numpy()
                    Generator.train()

            eigen_freq = self._eigen_frequency(epoch)
            if epoch % eigen_freq == 0 or epoch == self.epochs - 1:
                weight_info["eigen"] = self._compute_eigen_data(Generator, epoch)

            self.epoch_sig.emit(epoch + 1, d_loss_real.item() + d_loss_fake.item() + g_loss.item(), -g_loss.item(), None, None, weight_info, output)
            self.msleep(self.speed)

    def _get_optimizer(self, model):
        opts = {"Adam": optim.Adam, "SGD": optim.SGD, "RMSprop": optim.RMSprop, "AdamW": optim.AdamW}
        kwargs = {"lr": self.lr, "weight_decay": self.wd}
        if self.optimizer == "SGD": kwargs["momentum"] = 0.9
        return opts.get(self.optimizer, optim.Adam)(model.parameters(), **kwargs)

    def _current_lr(self, optimizer): return optimizer.param_groups[0]["lr"]

    def _get_weight_stats(self, model, prefix: str = "") -> dict:
        stats = {}
        for name, param in model.named_parameters():
            if param.data.numel() > 0:
                stats[prefix + name + "_m"] = float(param.data.mean())
                stats[prefix + name + "_s"] = float(param.data.std())
        return stats