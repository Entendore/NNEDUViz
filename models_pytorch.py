"""PyTorch neural network model definitions with weight access methods."""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MLP(nn.Module):
    def __init__(self, layers: List[int] = (2, 64, 32, 2), dropout: float = 0.0):
        super().__init__()
        self.layer_sizes = list(layers)
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.ReLU())
                if dropout > 0:
                    modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)
        self._features = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        layers = list(self.net.children())
        for i, layer in enumerate(layers):
            h = layer(h)
            if isinstance(layer, nn.Linear) and i < len(layers) - 1:
                self._features = h
        return h

    def get_features(self, x: torch.Tensor) -> np.ndarray:
        _ = self(x)
        if self._features is not None:
            return self._features.detach().cpu().numpy()
        return x.detach().cpu().numpy()

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        """Return all 2D weight matrices with their names."""
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class CNN(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        return self.classifier(self.features(x))

    def get_features(self, x: torch.Tensor) -> np.ndarray:
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
        return self.features(x).flatten(1).detach().cpu().numpy()

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class RNN(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.fc(output)

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class LSTM(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.fc(output)

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, average_attn_weights=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, attn_weights = self.mha(x, x, x, need_weights=True)
        self.attention_weights = attn_weights.detach().cpu()
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model: int = 32, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, num_heads) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, 1)
        self.all_attn = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_encoding(self.embedding(x))
        self.all_attn = []
        for block in self.blocks:
            h = block(h)
            self.all_attn.append(block.attention_weights)
        return self.output_proj(h)

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_weight_matrices(self) -> Dict[str, torch.Tensor]:
        result = {}
        for name, param in self.named_parameters():
            if param.dim() >= 2 and "weight" in name:
                result[name] = param.data
        return result