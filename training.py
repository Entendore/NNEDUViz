"""Background training thread for PyTorch models."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PySide6.QtCore import QThread, Signal
from factory import simple_pca, confusion_matrix, grid_2d


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
            "mlp": self._train_classification,
            "cnn": self._train_classification,
            "rnn": self._train_sequence,
            "lstm": self._train_sequence,
            "transformer": self._train_sequence,
            "gan": self._train_gan,
        }
        trainer = trainer_map.get(self.ntype)
        if trainer:
            trainer()
        self.finished_sig.emit()

    def _wait_if_paused(self) -> bool:
        while self._paused:
            if not self._running:
                return False
            self.msleep(50)
        return self._running

    def _update_frequency(self, epoch: int) -> int:
        if epoch <= 5:
            return 1
        if epoch <= 20:
            return 2
        if epoch <= 100:
            return 5
        return 10

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _train_classification(self):
        model = self.model
        model.train()
        
        # Fixed capitalization to match ui.py setup() kwargs
        X_train = torch.FloatTensor(self.X_train)
        y_train = torch.LongTensor(self.Y_train)
        X_test = torch.FloatTensor(self.X_test) if self.X_test is not None else None
        y_test = torch.LongTensor(self.Y_test) if self.Y_test is not None else None
        n_classes = int(y_train.max().item()) + 1
        
        criterion = nn.CrossEntropyLoss()
        optimizer = self._get_optimizer(model)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, self.bs, shuffle=True)

        for epoch in range(self.epochs):
            if not self._wait_if_paused():
                break
            model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(dataloader)

            grad_norms = {name: float(param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None}
            weight_info = {"lr": self._current_lr(optimizer), "grad_norms": grad_norms}
            weight_info.update(self._get_weight_stats(model))

            with torch.no_grad():
                model.eval()
                train_preds = model(X_train).argmax(1)
                train_acc = float((train_preds == y_train).float().mean())
                
                test_loss = None
                test_acc = None
                if X_test is not None:
                    test_loss = float(criterion(model(X_test), y_test))
                    test_preds = model(X_test).argmax(1)
                    test_acc = float((test_preds == y_test).float().mean())
                
                model.train()

            output_data = self._get_classification_output(
                model, X_train, y_train, X_test, y_test, n_classes, epoch
            )
            
            self.epoch_sig.emit(
                epoch + 1, train_loss, train_acc, 
                test_loss, test_acc, weight_info, output_data
            )
            self.msleep(self.speed)

    def _get_classification_output(self, model, X_train, y_train, X_test, y_test, n_classes, epoch):
        freq = self._update_frequency(epoch)
        output = {"type": "boundary"}
        
        if epoch % freq == 0 or epoch == self.epochs - 1:
            with torch.no_grad():
                model.eval()
                
                gx, gy, grid_points = grid_2d(X_train[:, :2].numpy())
                grid_predictions = model(torch.FloatTensor(grid_points)).argmax(1).numpy()
                
                output.update(
                    gx=gx.ravel(), gy=gy.ravel(), gp=grid_predictions,
                    train_x=X_train[:, 0].numpy(), train_y=y_train.numpy(),
                    dx=X_train[:, 0].numpy(), dy=X_train[:, 1].numpy(), dy2=y_train.numpy()
                )
                
                if X_test is not None:
                    output["test_x"] = X_test[:, 0].numpy()
                    output["test_y"] = y_test.numpy()
                
                if X_test is not None:
                    output["confusion"] = {
                        "true": y_test.numpy(), 
                        "pred": model(X_test).argmax(1).numpy(), 
                        "nc": n_classes,
                        "is_test": True
                    }
                    try:
                        features = model.get_features(X_test)
                        pca_features = simple_pca(features)
                        output["features"] = {
                            "x": pca_features[:, 0], "y": pca_features[:, 1], 
                            "labels": y_test.numpy(), "is_test": True
                        }
                    except Exception:
                        pass
                else:
                    output["confusion"] = {
                        "true": y_train.numpy(), 
                        "pred": model(X_train).argmax(1).numpy(), 
                        "nc": n_classes, "is_test": False
                    }
                
                model.train()
        return output

    def _train_sequence(self):
        model = self.model
        model.train()
        
        # Fixed capitalization to match ui.py setup() kwargs
        X_train = torch.FloatTensor(self.X_train)
        y_train = torch.FloatTensor(self.Y_train)
        X_test = torch.FloatTensor(self.X_test) if self.X_test is not None else None
        y_test = torch.FloatTensor(self.Y_test) if self.Y_test is not None else None
        
        criterion = nn.MSELoss()
        optimizer = self._get_optimizer(model)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, self.bs, shuffle=True)

        for epoch in range(self.epochs):
            if not self._wait_if_paused():
                break
            model.train()
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(dataloader)

            grad_norms = {name: float(param.grad.norm()) for name, param in model.named_parameters() if param.grad is not None}
            weight_info = {"lr": self._current_lr(optimizer), "grad_norms": grad_norms}
            weight_info.update(self._get_weight_stats(model))

            freq = self._update_frequency(epoch)
            output = {"type": "seq"}
            
            with torch.no_grad():
                model.eval()
                train_preds = model(X_train)
                ss_res = torch.sum((train_preds - y_train) ** 2)
                ss_tot = torch.sum((y_train - y_train.mean()) ** 2)
                train_r2 = 1 - float(ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                test_loss = None
                test_r2 = None
                if X_test is not None:
                    test_loss = float(criterion(model(X_test), y_test))
                    test_preds = model(X_test)
                    ss_res_t = torch.sum((test_preds - y_test) ** 2)
                    ss_tot_t = torch.sum((y_test - y_test.mean()) ** 2)
                    test_r2 = 1 - float(ss_res_t / ss_tot_t) if ss_tot_t > 0 else 0.0
                
                model.train()
                
                if epoch % freq == 0 or epoch == self.epochs - 1:
                    output["act"] = y_train[:3].numpy()
                    output["pred"] = train_preds[:3].numpy()
                    
                    if X_test is not None:
                        output["test_act"] = y_test[:3].numpy()
                        output["test_pred"] = model(X_test)[:3].numpy()
                
                if self.ntype == "transformer" and hasattr(model, "all_attn"):
                    if model.all_attn and epoch % freq == 0:
                        output["attention"] = [w[0].numpy() for w in model.all_attn if w is not None]

            self.epoch_sig.emit(
                epoch + 1, train_loss, train_r2,
                test_loss, test_r2, weight_info, output
            )
            self.msleep(self.speed)

    def _train_gan(self):
        Generator, Discriminator = self.model
        real_data = torch.FloatTensor(self.gd)
        n_samples = len(real_data)
        latent_dim = 16
        opt_G = optim.Adam(Generator.parameters(), lr=self.lr)
        opt_D = optim.Adam(Discriminator.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        for epoch in range(self.epochs):
            if not self._wait_if_paused():
                break
            Discriminator.train()
            opt_D.zero_grad()
            idx = np.random.choice(n_samples, self.bs, replace=False)
            real_batch = real_data[idx]
            real_loss = criterion(Discriminator(real_batch), torch.ones(self.bs, 1))
            fake_samples = Generator(torch.randn(self.bs, latent_dim)).detach()
            fake_loss = criterion(Discriminator(fake_samples), torch.zeros(self.bs, 1))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            opt_D.step()

            Generator.train()
            opt_G.zero_grad()
            fake_samples = Generator(torch.randn(self.bs, latent_dim))
            g_loss = criterion(Discriminator(fake_samples), torch.ones(self.bs, 1))
            g_loss.backward()
            opt_G.step()

            grad_norms = {}
            for name, param in Generator.named_parameters():
                if param.grad is not None:
                    grad_norms["G_" + name] = float(param.grad.norm())
            for name, param in Discriminator.named_parameters():
                if param.grad is not None:
                    grad_norms["D_" + name] = float(param.grad.norm())
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

            self.epoch_sig.emit(
                epoch + 1, d_loss.item() + g_loss.item(), -g_loss.item(),
                None, None, weight_info, output
            )
            self.msleep(self.speed)

    def _get_optimizer(self, model):
        if self.optimizer == "Adam":
            return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "SGD":
            return optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=0.9)
        elif self.optimizer == "RMSprop":
            return optim.RMSprop(model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == "AdamW":
            return optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)

    def _current_lr(self, optimizer):
        return optimizer.param_groups[0]["lr"]

    def _get_weight_stats(self, model, prefix: str = "") -> dict:
        stats = {}
        for name, param in model.named_parameters():
            if param.data.numel() > 0:
                stats[prefix + name + "_m"] = float(param.data.mean())
                stats[prefix + name + "_s"] = float(param.data.std())
        return stats