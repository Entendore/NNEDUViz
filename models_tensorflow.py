"""TensorFlow model definitions for all 6 architectures."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import numpy as np

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    TF_AVAILABLE = True
except Exception:
    pass

if TF_AVAILABLE:
    class TF_PosEnc(tf.keras.layers.Layer):
        def __init__(self, d, maxl=200, **kw):
            super().__init__(**kw)
            self.d = d
            self.maxl = maxl

        def build(self, bs):
            pe = np.zeros((self.maxl, self.d))
            p = np.arange(self.maxl)[:, None].astype(np.float32)
            d2 = np.exp(np.arange(0, self.d, 2, dtype=np.float32) * -(math.log(10000.0) / self.d))
            pe[:, 0::2] = np.sin(p * d2)
            pe[:, 1::2] = np.cos(p * d2)
            self.pe = tf.constant(pe[None])
            self.built = True

        def call(self, x):
            return x + self.pe[:, : tf.shape(x)[1]]

    class TF_MLP(tf.keras.Model):
        def __init__(self, layers=(2, 64, 32, 2), dropout=0.0):
            super().__init__()
            self.ls = list(layers)
            self.blocks = []
            for i in range(len(layers) - 1):
                blk = [tf.keras.layers.Dense(layers[i + 1])]
                if i < len(layers) - 2:
                    blk.append(tf.keras.layers.ReLU())
                    if dropout > 0:
                        blk.append(tf.keras.layers.Dropout(dropout))
                self.blocks.append(blk)
            self._feat = None

        def call(self, x, training=False):
            for i, blk in enumerate(self.blocks):
                for layer in blk:
                    x = layer(x, training=training) if isinstance(layer, tf.keras.layers.Dropout) else layer(x)
                if i < len(self.blocks) - 1:
                    self._feat = x
            return x

        def get_features(self, x):
            _ = self(x, training=False)
            return self._feat.numpy() if self._feat is not None else x.numpy()

    class TF_CNN(tf.keras.Model):
        def __init__(self, dropout=0.0):
            super().__init__()
            self.conv1 = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")
            self.pool1 = tf.keras.layers.MaxPooling1D(2)
            self.drop1 = tf.keras.layers.Dropout(dropout)
            self.conv2 = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")
            self.pool2 = tf.keras.layers.MaxPooling1D(2)
            self.flatten = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(64, activation="relu")
            self.drop2 = tf.keras.layers.Dropout(dropout)
            self.dense2 = tf.keras.layers.Dense(3)
            self._feat = None

        def call(self, x, training=False):
            x = self.pool1(self.conv1(x))
            x = self.drop1(x, training=training)
            x = self.pool2(self.conv2(x))
            self._feat = self.flatten(x)
            x = self.drop2(self.dense1(self._feat), training=training)
            return self.dense2(x)

        def get_features(self, x):
            _ = self(x, training=False)
            return self._feat.numpy()

    class TF_RNN(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.rnn = tf.keras.layers.SimpleRNN(32, return_sequences=True)
            self.fc = tf.keras.layers.Dense(1)

        def call(self, x):
            return self.fc(self.rnn(x))

    class TF_LSTM(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.lstm = tf.keras.layers.LSTM(32, return_sequences=True)
            self.fc = tf.keras.layers.Dense(1)

        def call(self, x):
            return self.fc(self.lstm(x))

    class TF_AttnBlock(tf.keras.layers.Layer):
        def __init__(self, d, nh, **kw):
            super().__init__(**kw)
            self.mha = tf.keras.layers.MultiHeadAttention(nh, d)
            self.n1 = tf.keras.layers.LayerNormalization()
            self.n2 = tf.keras.layers.LayerNormalization()
            self.aw = None

        def build(self, bs):
            self.ff = tf.keras.Sequential([
                tf.keras.layers.Dense(self.mha._key_dim * 2, activation="relu"),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(self.mha._key_dim),
            ])
            self.built = True

        def call(self, x):
            o, aw = self.mha(x, x, x, return_attention_scores=True)
            self.aw = aw
            h = self.n1(x + o)
            return self.n2(h + self.ff(h))

    class TF_Transformer(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.emb = tf.keras.layers.Dense(32)
            self.pos = TF_PosEnc(32)
            self.blocks = [TF_AttnBlock(32, 4), TF_AttnBlock(32, 4)]
            self.dec = tf.keras.layers.Dense(1)

        def call(self, x):
            h = self.pos(self.emb(x))
            for b in self.blocks:
                h = b(h)
            return self.dec(h)

        def get_attention(self):
            return [b.aw.numpy() if b.aw is not None else None for b in self.blocks]

    class TF_Gen(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(2),
            ])

        def call(self, x):
            return self.net(x)

    class TF_Dis(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.net = tf.keras.Sequential([
                tf.keras.layers.Dense(64), tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dense(64), tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])

        def call(self, x):
            return self.net(x)