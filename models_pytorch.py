"""PyTorch model definitions for all 6 architectures."""

import math
import torch
import torch.nn as nn


class PosEnc(nn.Module):
    def __init__(self, d, maxl=200):
        super().__init__()
        pe = torch.zeros(maxl, d)
        p = torch.arange(maxl).unsqueeze(1).float()
        d2 = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(p * d2)
        pe[:, 1::2] = torch.cos(p * d2)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PT_MLP(nn.Module):
    def __init__(self, layers=(2, 64, 32, 2), dropout=0.0):
        super().__init__()
        self.ls = list(layers)
        m = []
        for i in range(len(layers) - 1):
            m.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*m)
        self._feat = None

    def forward(self, x):
        h = x
        layers = list(self.net.children())
        for i, layer in enumerate(layers):
            h = layer(h)
            if isinstance(layer, nn.Linear) and i < len(layers) - 1:
                if i + 1 < len(layers) and not isinstance(layers[i + 1], (nn.ReLU, nn.Dropout)):
                    self._feat = h
                elif isinstance(layers[i + 1], nn.ReLU):
                    self._feat = h
        return h

    def get_features(self, x):
        _ = self(x)
        return self._feat.detach().cpu().numpy() if self._feat is not None else x.detach().cpu().numpy()


class PT_CNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.cls = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.cls(self.features(x))

    def get_features(self, x):
        return self.features(x).flatten(1).detach().cpu().numpy()


class PT_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


class PT_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc(self.lstm(x)[0])


class PT_AttnBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, nh, batch_first=True, average_attn_weights=False)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d, d))
        self.aw = None

    def forward(self, x):
        o, w = self.mha(x, x, x, need_weights=True)
        self.aw = w.detach().cpu()
        h = self.n1(x + o)
        return self.n2(h + self.ff(h))


class PT_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(1, 32)
        self.pos = PosEnc(32)
        self.blocks = nn.ModuleList([PT_AttnBlock(32, 4) for _ in range(2)])
        self.dec = nn.Linear(32, 1)
        self.all_attn = []

    def forward(self, x):
        h = self.pos(self.emb(x))
        self.all_attn = []
        for b in self.blocks:
            h = b(h)
            self.all_attn.append(b.aw)
        return self.dec(h)


class PT_Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


class PT_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 64),
            nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)