#!/usr/bin/env python3
"""
Neural Network Training Visualizer — PyTorch & TensorFlow
6 Architectures | Dual Framework | Gradient Flow | Attention Heatmap
Confusion Matrix | Feature Space PCA | Training Log | Export
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys, math, csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    TF_AVAILABLE = True
except Exception:
    pass

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib; matplotlib.use('QtAgg')

# ═══════════════════════════════════════════════════════════════════════════════
#  THEME
# ═══════════════════════════════════════════════════════════════════════════════
class T:
    BG="#1e1e2e";S1="#2a2a3d";S2="#353550";PRI="#89b4fa";GRN="#a6e3a1"
    PEA="#fab387";RED="#f38ba8";YEL="#f9e2af";MAU="#cba6f7";TXT="#cdd6f4"
    SUB="#a6adc8";DIM="#6c7086";OVR="#45475a";TEAL="#94e2d5";SKY="#74c7ec"
    @staticmethod
    def qc(h,a=255):return QColor(int(h[1:3],16),int(h[3:5],16),int(h[5:7],16),a)
    @staticmethod
    def css():
        return f"""
        QWidget{{background:{T.BG};color:{T.TXT};font-family:'Segoe UI',sans-serif;font-size:12px}}
        QGroupBox{{border:1px solid {T.OVR};border-radius:8px;margin-top:14px;padding-top:18px;
                    font-weight:bold;color:{T.SUB}}}
        QGroupBox::title{{subcontrol-origin:margin;left:12px;padding:0 6px}}
        QPushButton{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;
                    padding:6px 10px;color:{T.TXT};font-weight:bold;min-height:16px;font-size:11px}}
        QPushButton:hover{{background:{T.OVR}}}
        QPushButton:pressed{{background:{T.PRI};color:{T.BG}}}
        QPushButton:disabled{{color:{T.DIM};border-color:{T.DIM}}}
        QPushButton#startBtn{{background:#2d5a3e;border-color:{T.GRN}}}
        QPushButton#startBtn:hover{{background:#3a7a52}}
        QPushButton#stopBtn{{background:#5a2d2d;border-color:{T.RED}}}
        QPushButton#stopBtn:hover{{background:#7a3a3a}}
        QPushButton#resetBtn{{background:#5a4a2d;border-color:{T.YEL}}}
        QPushButton#resetBtn:hover{{background:#7a6a3a}}
        QComboBox{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;padding:5px 8px;color:{T.TXT};font-size:11px}}
        QComboBox::drop-down{{border:none;width:18px}}
        QComboBox QAbstractItemView{{background:{T.S2};color:{T.TXT};selection-background-color:{T.PRI}}}
        QDoubleSpinBox,QSpinBox{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;padding:3px 6px;color:{T.TXT};font-size:11px}}
        QLineEdit{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;padding:4px 8px;color:{T.TXT};font-size:11px}}
        QLabel{{color:{T.SUB}}}QLabel#desc{{color:{T.DIM};font-size:10px;font-style:italic;padding:2px}}
        QCheckBox{{color:{T.SUB};spacing:6px}}
        QSlider::groove:horizontal{{height:4px;background:{T.OVR};border-radius:2px}}
        QSlider::handle:horizontal{{background:{T.PRI};width:14px;margin:-5px 0;border-radius:7px}}
        QTabWidget::pane{{border:1px solid {T.OVR};border-radius:6px;background:{T.S1}}}
        QTabBar::tab{{background:{T.S2};color:{T.SUB};padding:6px 12px;border-top-left-radius:6px;
                      border-top-right-radius:6px;margin-right:2px;font-size:10px}}
        QTabBar::tab:selected{{background:{T.S1};color:{T.PRI};border-bottom:2px solid {T.PRI}}}
        QTableWidget{{background:{T.S1};gridline-color:{T.OVR};color:{T.TXT};font-size:10px;
                      border:1px solid {T.OVR};border-radius:4px}}
        QTableWidget::item{{padding:2px}}
        QHeaderView::section{{background:{T.S2};color:{T.SUB};padding:4px;border:1px solid {T.OVR};font-size:10px}}
        QScrollBar:vertical{{background:{T.S1};width:8px;border:none}}
        QScrollBar::handle:vertical{{background:{T.OVR};border-radius:4px;min-height:20px}}
        QScrollArea{{border:none}}
        QStatusBar{{background:{T.S1};color:{T.DIM};font-size:10px;border-top:1px solid {T.OVR}}}
        """

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════════════════════════════
def make_circles(n=500,noise=0.1):
    a=np.random.uniform(0,2*np.pi,n);n2=n//2
    ri=0.5+np.random.randn(n2)*noise;ro=1.0+np.random.randn(n-n2)*noise
    Xi=np.c_[ri*np.cos(a[:n2]),ri*np.sin(a[:n2])];Xo=np.c_[ro*np.cos(a[n2:]),ro*np.sin(a[n2:])]
    X=np.vstack([Xi,Xo]);y=np.hstack([np.zeros(n2),np.ones(n-n2)])
    return np.float32(X),np.int64(y)

def make_spirals(n=500,noise=0.15):
    n2=n//2;t=np.linspace(0,3*np.pi,n2)
    X1=np.c_[np.cos(t)*t/3,np.sin(t)*t/3]+np.random.randn(n2,2)*noise
    X2=np.c_[np.cos(t+np.pi)*t/3,np.sin(t+np.pi)*t/3]+np.random.randn(n2,2)*noise
    return np.float32(np.vstack([X1,X2])),np.int64(np.hstack([np.zeros(n2),np.ones(n2)]))

def make_xor(n=500,noise=0.15):
    cs=[(-1,-1),(1,1),(-1,1),(1,-1)];X=[];Y=[]
    for i,(cx,cy) in enumerate(cs):
        ni=n//4;X.append(np.c_[np.full(ni,cx)+np.random.randn(ni)*noise,np.full(ni,cy)+np.random.randn(ni)*noise])
        Y.append(np.full(ni,i%2,dtype=int))
    return np.float32(np.vstack(X)),np.int64(np.concatenate(Y))

def make_moons(n=500,noise=0.15):
    n2=n//2;a=np.linspace(0,np.pi,n2)
    X1=np.c_[np.cos(a),np.sin(a)]+np.random.randn(n2,2)*noise
    X2=np.c_[1-np.cos(a),0.5-np.sin(a)]+np.random.randn(n2,2)*noise
    return np.float32(np.vstack([X1,X2])),np.int64(np.hstack([np.zeros(n2),np.ones(n2)]))

def make_sine(sl=30,n=300):
    X,Y=[],[]
    for _ in range(n):
        ph=np.random.uniform(0,2*np.pi);fr=np.random.uniform(0.5,1.5)
        t=np.linspace(0,4*np.pi,sl+1);s=np.sin(fr*t+ph)
        X.append(s[:-1]);Y.append(s[1:])
    return np.float32(np.array(X))[...,None],np.float32(np.array(Y))[...,None]

def make_gan_data(n=500,mode='ring'):
    if mode=='ring':
        a=np.linspace(0,2*np.pi,n,endpoint=False)+np.random.randn(n)*0.1
        r=1+np.random.randn(n)*0.08;return np.float32(np.c_[r*np.cos(a),r*np.sin(a)])
    elif mode=='spiral':
        t=np.linspace(0,3*np.pi,n);d=np.c_[t*np.cos(t),t*np.sin(t)]*0.1+np.random.randn(n,2)*0.05
        return np.float32((d-d.mean(0))/d.std(0))
    else:
        cs=[(-1,-1),(1,1),(-1,1),(1,-1)];X=[]
        for i in range(n):X.append(cs[i%4]+np.random.randn(2)*0.3)
        return np.float32(np.array(X))

def make_signals(n=500,sl=32):
    X,Y=[],[]
    for _ in range(n):
        c=np.random.randint(3);t=np.linspace(0,4*np.pi,sl)
        if c==0:s=np.sin(t)
        elif c==1:s=np.sin(2*t)
        else:s=np.sin(t)*np.cos(t)
        X.append(s+np.random.randn(sl)*0.2);Y.append(c)
    return np.float32(np.array(X))[...,None],np.int64(np.array(Y))

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def simple_pca(X,nc=2):
    c=X-X.mean(0);_,_,Vt=np.linalg.svd(c,full_matrices=False);return c@Vt[:nc].T

def conf_matrix(yt,yp,nc):
    cm=np.zeros((nc,nc),dtype=int)
    for t,p in zip(yt,yp):cm[int(t)][int(p)]+=1
    return cm

def to_numpy(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
    if isinstance(x,tf.Tensor):return x.numpy()
    return np.array(x)

# ═══════════════════════════════════════════════════════════════════════════════
#  PYTORCH MODELS
# ═══════════════════════════════════════════════════════════════════════════════
class PosEnc(nn.Module):
    def __init__(self,d,maxl=200):
        super().__init__();pe=torch.zeros(maxl,d);p=torch.arange(maxl).unsqueeze(1).float()
        d2=torch.exp(torch.arange(0,d,2).float()*-(math.log(10000.)/d))
        pe[:,0::2]=torch.sin(p*d2);pe[:,1::2]=torch.cos(p*d2);self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):return x+self.pe[:,:x.size(1)]

class PT_MLP(nn.Module):
    def __init__(self,layers=[2,64,32,2],dropout=0.0):
        super().__init__();self.ls=layers;m=[]
        for i in range(len(layers)-1):
            m.append(nn.Linear(layers[i],layers[i+1]))
            if i<len(layers)-2:m.append(nn.ReLU());m.append(nn.Dropout(dropout))
        self.net=nn.Sequential(*m);self._f=None
    def forward(self,x):
        h=x;layers=list(self.net.children())
        for i,layer in enumerate(layers):
            h=layer(h)
            if isinstance(layer,nn.Linear) and i<len(layers)-1 and layers[i+1] not in (nn.ReLU(),nn.Dropout()):self._f=h
        return h
    def get_features(self,x):_=self(x);return self._f.detach().cpu().numpy() if self._f is not None else x.detach().cpu().numpy()

class PT_CNN(nn.Module):
    def __init__(self,dropout=0.0):
        super().__init__()
        self.features=nn.Sequential(nn.Conv1d(1,16,5,padding=2),nn.ReLU(),nn.MaxPool1d(2),
                                     nn.Dropout(dropout),nn.Conv1d(16,32,5,padding=2),nn.ReLU(),nn.MaxPool1d(2))
        self.cls=nn.Sequential(nn.Flatten(),nn.Linear(32*8,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,3))
    def forward(self,x):return self.cls(self.features(x))
    def get_features(self,x):return self.features(x).flatten(1).detach().cpu().numpy()

class PT_RNN(nn.Module):
    def __init__(self):super().__init__();self.rnn=nn.RNN(1,32,1,batch_first=True);self.fc=nn.Linear(32,1)
    def forward(self,x):return self.fc(self.rnn(x)[0])

class PT_LSTM(nn.Module):
    def __init__(self):super().__init__();self.lstm=nn.LSTM(1,32,1,batch_first=True);self.fc=nn.Linear(32,1)
    def forward(self,x):return self.fc(self.lstm(x)[0])

class PT_AttnBlock(nn.Module):
    def __init__(self,d,nh):
        super().__init__()
        self.mha=nn.MultiheadAttention(d,nh,batch_first=True,average_attn_weights=False)
        self.n1=nn.LayerNorm(d);
        self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*2),nn.ReLU(),nn.Dropout(0.1),nn.Linear(d,d));self.aw=None
    def forward(self,x):
        o,w=self.mha(x,x,x,need_weights=True);self.aw=w.detach().cpu()
        h=self.n1(x+o);return self.n2(h+self.ff(h))

class PT_Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Linear(1,32);
        self.pos=PosEnc(32)
        self.blocks=nn.ModuleList([PT_AttnBlock(32,4) for _ in range(2)])
        self.dec=nn.Linear(32,1);
        self.all_attn=[]
    def forward(self,x):
        h=self.pos(self.emb(x));self.all_attn=[]
        for b in self.blocks:h=b(h);self.all_attn.append(b.aw)
        return self.dec(h)

class PT_Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    def forward(self,x):return self.net(x)

class PT_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,64),nn.LeakyReLU(0.2),nn.Linear(64,64),nn.LeakyReLU(0.2),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x):return self.net(x)

# ═══════════════════════════════════════════════════════════════════════════════
#  TENSORFLOW MODELS
# ═══════════════════════════════════════════════════════════════════════════════
if TF_AVAILABLE:
    class TF_PosEnc(tf.keras.layers.Layer):
        def __init__(self,d,maxl=200,**kw):super().__init__(**kw);self.d=d;self.maxl=maxl
        def build(self,bs):
            pe=np.zeros((self.maxl,self.d));p=np.arange(self.maxl)[:,None].astype(np.float32)
            d2=np.exp(np.arange(0,self.d,2,dtype=np.float32)*-(math.log(10000.)/self.d))
            pe[:,0::2]=np.sin(p*d2);pe[:,1::2]=np.cos(p*d2)
            self.pe=tf.constant(pe[None]);self.built=True
        def call(self,x):return x+self.pe[:,:tf.shape(x)[1]]

    class TF_MLP(tf.keras.Model):
        def __init__(self,layers=[2,64,32,2],dropout=0.0):
            super().__init__();self.ls=layers;self.blocks=[]
            for i in range(len(layers)-1):
                d=tf.keras.layers.Dense(layers[i+1])
                blk=[d]
                if i<len(layers)-2:
                    blk.append(tf.keras.layers.ReLU())
                    if dropout>0:blk.append(tf.keras.layers.Dropout(dropout))
                self.blocks.append(blk)
            self._feat=None
        def call(self,x,training=False):
            for i,blk in enumerate(self.blocks):
                for layer in blk:
                    x=layer(x,training=training) if isinstance(layer,tf.keras.layers.Dropout) else layer(x)
                if i<len(self.blocks)-1:self._feat=x
            return x
        def get_features(self,x):_=self(x,training=False);return self._feat.numpy() if self._feat is not None else x.numpy()

    class TF_CNN(tf.keras.Model):
        def __init__(self,dropout=0.0):
            super().__init__()
            self.conv1=tf.keras.layers.Conv1D(16,5,padding='same',activation='relu')
            self.pool1=tf.keras.layers.MaxPooling1D(2)
            self.drop1=tf.keras.layers.Dropout(dropout)
            self.conv2=tf.keras.layers.Conv1D(32,5,padding='same',activation='relu')
            self.pool2=tf.keras.layers.MaxPooling1D(2)
            self.flatten=tf.keras.layers.Flatten()
            self.dense1=tf.keras.layers.Dense(64,activation='relu')
            self.drop2=tf.keras.layers.Dropout(dropout)
            self.dense2=tf.keras.layers.Dense(3)
            self._feat=None
        def call(self,x,training=False):
            x=self.pool1(self.conv1(x));x=self.drop1(x,training=training)
            x=self.pool2(self.conv2(x))
            self._feat=self.flatten(x)
            x=self.drop2(self.dense1(self._feat),training=training)
            return self.dense2(x)
        def get_features(self,x):_=self(x,training=False);return self._feat.numpy()

    class TF_RNN(tf.keras.Model):
        def __init__(self):super().__init__();self.rnn=tf.keras.layers.SimpleRNN(32,return_sequences=True);self.fc=tf.keras.layers.Dense(1)
        def call(self,x):return self.fc(self.rnn(x))

    class TF_LSTM(tf.keras.Model):
        def __init__(self):super().__init__();self.lstm=tf.keras.layers.LSTM(32,return_sequences=True);self.fc=tf.keras.layers.Dense(1)
        def call(self,x):return self.fc(self.lstm(x))

    class TF_AttnBlock(tf.keras.layers.Layer):
        def __init__(self,d,nh,**kw):
            super().__init__(**kw)
            self.mha=tf.keras.layers.MultiHeadAttention(nh,d);self.n1=tf.keras.layers.LayerNormalization()
            self.n2=tf.keras.layers.LayerNormalization();self.aw=None
        def build(self,bs):
            self.ff=tf.keras.Sequential([tf.keras.layers.Dense(self.mha._key_dim*2,activation='relu'),
                                         tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(self.mha._key_dim)])
            self.built=True
        def call(self,x):
            o,aw=self.mha(x,x,x,return_attention_scores=True);self.aw=aw;h=self.n1(x+o)
            return self.n2(h+self.ff(h))

    class TF_Transformer(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.emb=tf.keras.layers.Dense(32)
            self.pos=TF_PosEnc(32)
            self.blocks=[TF_AttnBlock(32,4),TF_AttnBlock(32,4)]
            self.dec=tf.keras.layers.Dense(1)
        def call(self,x):h=self.pos(self.emb(x));h=self.blocks[0](h);h=self.blocks[1](h);return self.dec(h)
        def get_attention(self):
            return [b.aw.numpy() if b.aw is not None else None for b in self.blocks]

    class TF_Gen(tf.keras.Model):
        def __init__(self):
          super().__init__()
          self.net=tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(2)])
        def call(self,x):return self.net(x)

    class TF_Dis(tf.keras.Model):
        def __init__(self):
          super().__init__()
          self.net=tf.keras.Sequential([tf.keras.layers.Dense(64),tf.keras.layers.LeakyReLU(0.2),
                                          tf.keras.layers.Dense(64),tf.keras.layers.LeakyReLU(0.2),
                                          tf.keras.layers.Dense(1,activation='sigmoid')])
        def call(self,x):return self.net(x)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════════════
def make_model(ntype,framework,layers=None,dropout=0.0):
    if framework=='pytorch':
        if ntype=='mlp':return PT_MLP(layers or[2,64,32,2],dropout)
        elif ntype=='cnn':return PT_CNN(dropout)
        elif ntype=='rnn':return PT_RNN()
        elif ntype=='lstm':return PT_LSTM()
        elif ntype=='transformer':return PT_Transformer()
        elif ntype=='gan':return PT_Gen(),PT_Dis()
    elif framework=='tensorflow' and TF_AVAILABLE:
        if ntype=='mlp':return TF_MLP(layers or[2,64,32,2],dropout)
        elif ntype=='cnn':return TF_CNN(dropout)
        elif ntype=='rnn':return TF_RNN()
        elif ntype=='lstm':return TF_LSTM()
        elif ntype=='transformer':return TF_Transformer()
        elif ntype=='gan':return TF_Gen(),TF_Dis()
    return None

def count_params(model,framework):
    if framework=='pytorch':return sum(p.numel() for p in model.parameters())
    elif framework=='tensorflow':return sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    return 0

# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING THREAD
# ═══════════════════════════════════════════════════════════════════════════════
class TrainThread(QThread):
    epoch_sig=Signal(int,float,float,dict,object)
    finished_sig=Signal()
    def __init__(self):super().__init__();self._run=False;self._pause=False;self.speed=5
    def setup(self,**kw):self.__dict__.update(kw)
    def run(self):self._run=True;self._pause=False;getattr(self,f'_train_{self.ntype}')();self.finished_sig.emit()
    def _wait(self):
        while self._pause:
            if not self._run:return False
            self.msleep(50)
        return self._run
    def _freq(self,ep):
        if ep<=5:return 1
        if ep<=20:return 2
        if ep<=100:return 5
        return 10
    def stop(self):self._run=False
    def pause(self):self._pause=True
    def resume(self):self._pause=False

    # ── CLASSIFICATION ──────────────────────────────────────────────────────
    def _train_cls(self):
        if self.fw=='pytorch':self._train_cls_pt()
        else:self._train_cls_tf()

    def _train_cls_pt(self):
        m=self.model;m.train();X=torch.FloatTensor(self.Xd);y=torch.LongTensor(self.Yd)
        nc=int(y.max().item()+1);crit=nn.CrossEntropyLoss()
        opt=optim.Adam(m.parameters(),lr=self.lr,weight_decay=self.wd)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,y),self.bs,True)
        for ep in range(self.epochs):
            if not self._wait():break
            m.train();tl=0
            for bx,by in dl:opt.zero_grad();l=crit(m(bx),by);l.backward();opt.step();tl+=l.item()
            al=tl/len(dl);gn={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
            wi={'lr':self.lr,'grad_norms':gn};wi.update(self._wi_pt(m))
            od=self._make_cls_od(m,X,y,nc,ep);self.epoch_sig.emit(ep+1,al,float((m(X).argmax(1)==y).float().mean()),wi,od);self.msleep(self.speed)

    def _train_cls_tf(self):
        m=self.model;Xd=self.Xd;Yd=self.Yd;nc=int(Yd.max()+1)
        m.build((None,)+Xd.shape[1:])
        opt=tf.keras.optimizers.Adam(learning_rate=self.lr)
        crit=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ds=tf.data.Dataset.from_tensor_slices((Xd,Yd)).shuffle(len(Xd)).batch(self.bs)
        for ep in range(self.epochs):
            if not self._wait():break
            tl=0
            for bx,by in ds:
                with tf.GradientTape() as tape:
                    l=crit(by,m(bx,training=True))
                grads=tape.gradient(l,m.trainable_variables);opt.apply_gradients(zip(grads,m.trainable_variables));tl+=float(l)
            al=tl/max(1,len(Xd)//self.bs)
            gn={f'layer{i}':float(tf.norm(g).numpy()) for i,g in enumerate(grads) if g is not None}
            wi={'lr':float(opt.learning_rate.numpy()),'grad_norms':gn};wi.update(self._wi_tf(m))
            preds=np.argmax(m(Xd,training=False).numpy(),axis=1)
            acc=float((preds==Yd).mean())
            od=self._make_cls_od_tf(m,Xd,Yd,nc,ep)
            self.epoch_sig.emit(ep+1,al,acc,wi,od);self.msleep(self.speed)

    def _make_cls_od(self,m,X,y,nc,ep):
        f=self._freq(ep);od={'type':'boundary'}
        if ep%f==0 or ep==self.epochs-1:
            with torch.no_grad():
                m.eval();r=60;xn,xx=X[:,0].min()-.5,X[:,0].max()+.5;yn,yx=X[:,1].min()-.5,X[:,1].max()+.5
                gx,gy=np.meshgrid(np.linspace(xn,xx,r),np.linspace(yn,yx,r))
                g=torch.FloatTensor(np.c_[gx.ravel(),gy.ravel()]);gp=m(g).argmax(1).numpy()
                od.update({'gx':gx.ravel(),'gy':gy.ravel(),'gp':gp,'dx':X[:,0].numpy(),'dy':X[:,1].numpy(),'dy2':y.numpy()})
                p=m(X).argmax(1);m.train()
                od['confusion']={'true':y.numpy(),'pred':p.numpy(),'nc':nc}
                try:f2=m.get_features(X);fp=simple_pca(f2);od['features']={'x':fp[:,0],'y':fp[:,1],'labels':y.numpy()}
                except:pass
        return od

    def _make_cls_od_tf(self,m,Xd,Yd,nc,ep):
        f=self._freq(ep);od={'type':'boundary'}
        if ep%f==0 or ep==self.epochs-1:
            r=60;xn,xx=Xd[:,0].min()-.5,Xd[:,0].max()+.5;yn,yx=Xd[:,1].min()-.5,Xd[:,1].max()+.5
            gx,gy=np.meshgrid(np.linspace(xn,xx,r),np.linspace(yn,yx,r))
            gp=np.argmax(m(np.c_[gx.ravel(),gy.ravel()].astype(np.float32),training=False).numpy(),axis=1)
            od.update({'gx':gx.ravel(),'gy':gy.ravel(),'gp':gp,'dx':Xd[:,0],'dy':Xd[:,1],'dy2':Yd})
            preds=np.argmax(m(Xd,training=False).numpy(),axis=1)
            od['confusion']={'true':Yd,'pred':preds,'nc':nc}
            try:f2=m.get_features(Xd.astype(np.float32));fp=simple_pca(f2);od['features']={'x':fp[:,0],'y':fp[:,1],'labels':Yd}
            except:pass
        return od

    # ── SEQUENCE ───────────────────────────────────────────────────────────
    def _train_seq(self):
        if self.fw=='pytorch':self._train_seq_pt()
        else:self._train_seq_tf()

    def _train_seq_pt(self):
        m=self.model;m.train();X=torch.FloatTensor(self.Xd);y=torch.FloatTensor(self.Yd)
        crit=nn.MSELoss();opt=optim.Adam(m.parameters(),lr=self.lr,weight_decay=self.wd)
        dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,y),self.bs,True)
        for ep in range(self.epochs):
            if not self._wait():break
            m.train();tl=0
            for bx,by in dl:opt.zero_grad();l=crit(m(bx),by);l.backward();opt.step();tl+=l.item()
            al=tl/len(dl)
            gn={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.grad is not None}
            wi={'lr':self.lr,'grad_norms':gn};wi.update(self._wi_pt(m))
            f=self._freq(ep);od={'type':'seq'}
            with torch.no_grad():
                m.eval();p=m(X);met=1-float(torch.mean((p-y)**2));m.train()
                if ep%f==0 or ep==self.epochs-1:od['act']=y[:3].numpy();od['pred']=p[:3].numpy()
                if self.ntype=='transformer' and hasattr(m,'all_attn') and m.all_attn:
                    if ep%f==0:od['attention']=[w[0].numpy() for w in m.all_attn if w is not None]
            self.epoch_sig.emit(ep+1,al,met,wi,od);self.msleep(self.speed)

    def _train_seq_tf(self):
        m=self.model;Xd=self.Xd;Yd=self.Yd
        m.build((None,)+Xd.shape[1:])
        opt=tf.keras.optimizers.Adam(learning_rate=self.lr);crit=tf.keras.losses.MeanSquaredError()
        ds=tf.data.Dataset.from_tensor_slices((Xd,Yd)).shuffle(len(Xd)).batch(self.bs)
        for ep in range(self.epochs):
            if not self._wait():break
            tl=0
            for bx,by in ds:
                with tf.GradientTape() as tape:l=crit(by,m(bx,training=True))
                grads=tape.gradient(l,m.trainable_variables);opt.apply_gradients(zip(grads,m.trainable_variables));tl+=float(l)
            al=tl/max(1,len(Xd)//self.bs)
            gn={f'layer{i}':float(tf.norm(g).numpy()) for i,g in enumerate(grads) if g is not None}
            wi={'lr':float(opt.learning_rate.numpy()),'grad_norms':gn};wi.update(self._wi_tf(m))
            p=m(Xd,training=False).numpy();met=1-float(np.mean((p-Yd)**2))
            f=self._freq(ep);od={'type':'seq'}
            if ep%f==0 or ep==self.epochs-1:od['act']=Yd[:3];od['pred']=p[:3]
            if self.ntype=='transformer' and hasattr(m,'get_attention'):
                if ep%f==0:
                    _=m(Xd[:1],training=False)
                    attn=m.get_attention()
                    od['attention']=[a[0] for a in attn if a is not None]
            self.epoch_sig.emit(ep+1,al,met,wi,od);self.msleep(self.speed)

    def _train_mlp(self):self._train_cls()
    def _train_cnn(self):self._train_cls()
    def _train_rnn(self):self._train_seq()
    def _train_lstm(self):self._train_seq()
    def _train_transformer(self):self._train_seq()

    # ── GAN ────────────────────────────────────────────────────────────────
    def _train_gan(self):
        if self.fw=='pytorch':self._train_gan_pt()
        else:self._train_gan_tf()

    def _train_gan_pt(self):
        G,D=self.model;rd=torch.FloatTensor(self.gd);n=len(rd);ld=16
        oG=optim.Adam(G.parameters(),lr=self.lr);oD=optim.Adam(D.parameters(),lr=self.lr)
        crit=nn.BCELoss()
        for ep in range(self.epochs):
            if not self._wait():break
            D.train();oD.zero_grad()
            idx=np.random.choice(n,self.bs,replace=False);real=rd[idx]
            fake=G(torch.randn(self.bs,ld)).detach()
            dl=crit(D(real),torch.ones(self.bs,1))+crit(D(fake),torch.zeros(self.bs,1))
            dl.backward();oD.step()
            G.train();oG.zero_grad()
            fake=G(torch.randn(self.bs,ld));gl=crit(D(fake),torch.ones(self.bs,1));gl.backward();oG.step()
            gn={};G.train()
            for n2,p2 in G.named_parameters():
                if p2.grad is not None:gn['G_'+n2]=float(p2.grad.norm())
            for n2,p2 in D.named_parameters():
                if p2.grad is not None:gn['D_'+n2]=float(p2.grad.norm())
            wi={'lr':self.lr,'grad_norms':gn};wi.update(self._wi_pt(G,'G_'));wi.update(self._wi_pt(D,'D_'))
            od={'type':'gan','real':rd.numpy()};f=self._freq(ep)
            if ep%f==0 or ep==self.epochs-1:
                with torch.no_grad():G.eval();od['gen']=G(torch.randn(200,ld)).numpy();G.train()
            self.epoch_sig.emit(ep+1,dl.item()+gl.item(),-gl.item(),wi,od);self.msleep(self.speed)

    def _train_gan_tf(self):
        G,D=self.model;gd=self.gd;n=len(gd);ld=16
        oG=tf.keras.optimizers.Adam(learning_rate=self.lr);oD=tf.keras.optimizers.Adam(learning_rate=self.lr)
        bc=tf.keras.losses.BinaryCrossentropy()
        G.build((None,ld));D.build((None,2))
        for ep in range(self.epochs):
            if not self._wait():break
            idx=np.random.choice(n,self.bs,replace=False);real=gd[idx]
            noise=np.random.randn(self.bs,ld).astype(np.float32)
            with tf.GradientTape() as tD:
                dl_r=bc(tf.ones((self.bs,1)),D(real,training=True))
                fake=G(noise,training=True);dl_f=bc(tf.zeros((self.bs,1)),D(fake,training=True))
                dloss=dl_r+dl_f
            dgrads=tD.gradient(dloss,D.trainable_variables);oD.apply_gradients(zip(dgrads,D.trainable_variables))
            noise=np.random.randn(self.bs,ld).astype(np.float32)
            with tf.GradientTape() as tG:
                fake=G(noise,training=True);gloss=bc(tf.ones((self.bs,1)),D(fake,training=True))
            ggrads=tG.gradient(gloss,G.trainable_variables);oG.apply_gradients(zip(ggrads,G.trainable_variables))
            gn={}
            for i,g in enumerate(dgrads):
                if g is not None:gn[f'D_l{i}']=float(tf.norm(g).numpy())
            for i,g in enumerate(ggrads):
                if g is not None:gn[f'G_l{i}']=float(tf.norm(g).numpy())
            wi={'lr':float(oG.learning_rate.numpy()),'grad_norms':gn};wi.update(self._wi_tf(G,'G_'));wi.update(self._wi_tf(D,'D_'))
            od={'type':'gan','real':gd};f=self._freq(ep)
            if ep%f==0 or ep==self.epochs-1:od['gen']=G(np.random.randn(200,ld).astype(np.float32),training=False).numpy()
            self.epoch_sig.emit(ep+1,float(dloss+gloss),-float(gloss),wi,od);self.msleep(self.speed)

    # ── WEIGHT INFO EXTRACTORS ─────────────────────────────────────────────
    def _wi_pt(self,m,p=''):
        info={}
        for n,p2 in m.named_parameters():
            if p2.data.numel()>0:
                info[p+n+'_m']=float(p2.data.mean());info[p+n+'_s']=float(p2.data.std())
        return info

    def _wi_tf(self,m,p=''):
        info={}
        for i,v in enumerate(m.trainable_variables):
            a=v.numpy();info[p+f'layer{i}_m']=float(np.mean(a));info[p+f'layer{i}_s']=float(np.std(a))
        return info

# ═══════════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE WIDGET
# ═══════════════════════════════════════════════════════════════════════════════
class ArchWidget(QWidget):
    def __init__(self):super().__init__();self.ntype='mlp';self.fw='pytorch';self.setMinimumHeight(170)
    def set_type(self,t,fw=None,layers=None):self.ntype=t;self.fw=fw or self.fw;self.layers=layers;self.update()
    def paintEvent(self,e):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.fillRect(self.rect(),T.qc(T.S1))
        getattr(self,f'_draw_{self.ntype}')(p,self.width(),self.height())
        # Framework badge
        badge="PyTorch" if self.fw=='pytorch' else "TensorFlow"
        bc=T.PRI if self.fw=='pytorch' else T.PEA
        self._rrect(p,self.width()-75,4,70,18,4,T.qc(bc,40),bc)
        f=p.font();f.setPixelSize(9);f.setBold(True);p.setFont(f);p.setPen(T.qc(bc))
        p.drawText(self.width()-75,16,badge);p.end()
    def _rrect(self,p,x,y,w,h,r,col,border=None):
        p.setBrush(T.qc(col));p.setPen(QPen(T.qc(border) if border else Qt.NoPen,1 if border else 0))
        p.drawRoundedRect(x,y,w,h,r,r)
    def _txt(self,p,x,y,t,c=T.TXT,sz=10,b=False):
        f=p.font();f.setPixelSize(sz);f.setBold(b);p.setFont(f);p.setPen(T.qc(c));p.drawText(x,y,t)
    def _arrow(self,p,x1,y1,x2,y2,c=T.OVR):
        p.setPen(QPen(T.qc(c),2));p.drawLine(x1,y1,x2,y2)
        a=math.atan2(y2-y1,x2-x1);al=7
        p.drawLine(x2,y2,x2-al*math.cos(a-.4),y2-al*math.sin(a-.4))
        p.drawLine(x2,y2,x2-al*math.cos(a+.4),y2-al*math.sin(a+.4))
    def _neurons(self,p,cx,cy,n,r,sp,c):
        ys=[cy-sp*(n-1)/2+i*sp for i in range(n)]
        for yy in ys:p.setBrush(T.qc(c,50));p.setPen(QPen(T.qc(c),1.5));p.drawEllipse(QPointF(cx,yy),r,r)
    def _arc(self,p,x1,y1,x2,y2,curv=18):
        path=QPainterPath();path.moveTo(x1,y1);mx=(x1+x2)/2
        path.cubicTo(mx,y1-curv,mx,y2-curv,x2,y2);p.drawPath(path)

    def _draw_mlp(self,p,w,h):
        ls=self.layers or[2,64,32,2];nl=len(ls)
        lx=[w*0.08+i*(w*0.84)/(nl-1) for i in range(nl)]
        nr=min(10,max(3,h*0.04));sp=min(14,(h-30)/max(ls))
        cs=[T.PRI]+[T.GRN]*(nl-2)+[T.PEA]
        for li,(n,x,c) in enumerate(zip(ls,lx,cs)):
            self._neurons(p,x,h//2,n,nr,sp,c)
            lb='Input' if li==0 else 'Output' if li==nl-1 else f'H{li}'
            self._txt(p,x-10,h-8,lb,c,8,True)
        for li in range(nl-1):
            sp1=sp
            for yi in range(ls[li]):
                y1=h//2-sp1*(ls[li]-1)/2+yi*sp1
                for yi2 in range(ls[li+1]):
                    y2=h//2-sp1*(ls[li+1]-1)/2+yi2*sp1
                    p.setPen(QPen(T.qc(T.OVR,40),0.5));p.drawLine(int(lx[li]+nr),int(y1),int(lx[li+1]-nr),int(y2))
        self._txt(p,8,12,"MLP — Multi-Layer Perceptron",T.PRI,11,True)
        self._txt(p,8,24,f"{' → '.join(map(str,ls))}",T.DIM,8)

    def _draw_cnn(self,p,w,h):
        bw,bh=75,46;y=h//2-bh//2
        xs=[w*0.04,w*0.18,w*0.29,w*0.43,w*0.54,w*0.71,w*0.82]
        cs=[T.PRI,T.GRN,T.DIM,T.GRN,T.DIM,T.MAU,T.PEA]
        ns=['Input\n1×32','Conv1D\n16ch','Pool','Conv1D\n32ch','Pool','FC 64','Out 3']
        for x,c,nm in zip(xs,cs,ns):
            self._rrect(p,x,y,bw,bh,6,c)
            for j,ln in enumerate(nm.split('\n')):self._txt(p,x+bw//2-len(ln)*3,y+bh//2-3+j*12,ln,T.TXT,8,j==0)
        for i in range(len(xs)-1):self._arrow(p,int(xs[i]+bw),int(y+bh//2),int(xs[i+1]),int(y+bh//2))
        self._txt(p,8,12,"CNN — 1D Convolutional Network",T.PRI,11,True)

    def _draw_rnn(self,p,w,h):
        ns=4;bw=68;bh=42;gap=(w-70)/(ns+1);cy=h//2
        for i in range(ns):
            x=45+i*gap;self._rrect(p,x,cy-bh//2,bw,bh,6,T.GRN)
            self._txt(p,x+bw//2-10,cy-2,"RNN",T.TXT,9,True);self._txt(p,x+bw//2-8,cy+11,f"h{i}",T.MAU,7)
            if i<ns-1:
                self._arrow(p,int(x+bw),int(cy-8),int(x+gap),int(cy-8),T.PRI)
                self._arrow(p,int(x+bw),int(cy+8),int(x+gap),int(cy+8),T.MAU)
            if i>0:p.setPen(QPen(T.qc(T.MAU,90),1.5,Qt.DashLine));self._arc(p,x-6,cy+4,x-6-gap+14,cy+4)
            self._txt(p,x+bw//2-4,cy-bh//2-11,f"x{i}",T.PRI,7);self._txt(p,x+bw//2-4,cy+bh//2+11,f"y{i}",T.PEA,7)
        self._txt(p,8,12,"RNN — Recurrent Neural Network (unrolled)",T.PRI,11,True)

    def _draw_lstm(self,p,w,h):
        ns=3;bw=88;bh=54;gap=(w-90)/(ns+1);cy=h//2
        gc,gn=[T.RED,T.YEL,T.TEAL,T.RED],['f','i','ĩ','o']
        for i in range(ns):
            x=55+i*gap;gw=17;gg=(bw-4*gw)/5
            self._rrect(p,x,cy-bh//2,bw,bh,6,T.S2,T.MAU)
            for gi,(c2,n2) in enumerate(zip(gc,gn)):
                gx=x+gg+gi*(gw+gg);self._rrect(p,gx,cy-bh//2+5,gw,11,3,c2)
                self._txt(p,gx+gw//2-2,cy-bh//2+14,n2,T.BG,7,True)
            self._txt(p,x+bw//2-8,cy+2,"LSTM",T.TXT,9,True)
            if i<ns-1:
                self._arrow(p,int(x+bw),int(cy-11),int(x+gap),int(cy-11),T.PRI)
                self._arrow(p,int(x+bw),int(cy+11),int(x+gap),int(cy+11),T.MAU)
                p.setPen(QPen(T.qc(T.TEAL,80),1.5,Qt.DashLine));self._arc(p,x-4,cy+15,x-4-gap+12,cy+15,18)
        self._txt(p,8,12,"LSTM — Long Short-Term Memory (unrolled)",T.PRI,11,True)

    def _draw_transformer(self,p,w,h):
        bw,bh=82,34;cy=h//2;gap=7
        blks=[("Input",T.PRI),("PosEnc",T.DIM),("MultiHead\nAttn",T.YEL),("Add&Norm",T.TEAL),("FFN",T.GRN),("Add&Norm",T.TEAL),("Output",T.PEA)]
        tw=len(blks)*(bw+gap)-gap;sx=(w-tw)//2;xs=[]
        for i,(nm,c) in enumerate(blks):
            x=sx+i*(bw+gap);xs.append(x);self._rrect(p,x,cy-bh//2,bw,bh,6,c)
            for j,ln in enumerate(nm.split('\n')):self._txt(p,x+bw//2-len(ln)*3,cy-1+j*11,ln,T.TXT,8,j==0)
            if i<len(blks)-1:self._arrow(p,int(x+bw),int(cy),int(xs[i+1]),int(cy))
        ry=cy+bh//2+5;p.setPen(QPen(T.qc(T.TEAL,70),1.5,Qt.DashLine))
        p.drawLine(int(xs[2]),int(ry),int(xs[4]),int(ry))
        self._txt(p,8,12,"Transformer — Encoder with Self-Attention",T.PRI,11,True)

    def _draw_gan(self,p,w,h):
        by1=h//2-48;by2=h//2+18;bw=72;bh=32;sx=w*0.06;gg=(w*0.40)/4
        gn2=[("z~N(0,1)",T.DIM),("FC 64",T.GRN),("FC 64",T.GRN),("FC 2",T.PEA)]
        dn=[("Data x",T.PRI),("FC 64",T.RED),("FC 64",T.RED),("σ",T.RED)]
        for i,(nm,c) in enumerate(gn2):
            x=sx+i*gg;self._rrect(p,x,by1,bw,bh,6,c);self._txt(p,x+bw//2-len(nm)*3,by1+bh//2+3,nm,T.TXT,8)
            if i<len(gn2)-1:self._arrow(p,int(x+bw),int(by1+bh//2),int(x+gg),int(by1+bh//2))
        for i,(nm,c) in enumerate(dn):
            x=sx+i*gg;self._rrect(p,x,by2,bw,bh,6,c);self._txt(p,x+bw//2-len(nm)*3,by2+bh//2+3,nm,T.TXT,8)
            if i<len(dn)-1:self._arrow(p,int(x+bw),int(by2+bh//2),int(x+gg),int(by2+bh//2))
        fx=sx+3*gg+bw+12;self._rrect(p,fx,by1,bw,bh,6,T.MAU);self._txt(p,fx+bw//2-18,by1+bh//2+3,"Fake Data",T.TXT,8)
        self._arrow(p,int(sx+3*gg+bw),int(by1+bh//2),int(fx),int(by1+bh//2))
        self._arrow(p,int(fx+bw//2),int(by1+bh),int(fx+bw//2),int(by2),T.MAU)
        self._arrow(p,int(fx),int(by2+bh//2),int(sx+gg),int(by2+bh//2))
        self._txt(p,w*0.50,by1-7,"GENERATOR",T.GRN,10,True);self._txt(p,w*0.50,by2-7,"DISCRIMINATOR",T.RED,10,True)
        self._txt(p,8,12,"GAN — Generative Adversarial Network",T.PRI,11,True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT CANVASES
# ═══════════════════════════════════════════════════════════════════════════════
class PlotCanvas(FigureCanvas):
    def __init__(self,title='',nrows=1,ncols=1):
        self.fig=Figure(figsize=(4,3),dpi=85);self.fig.patch.set_facecolor(T.S1);super().__init__(self.fig)
        if nrows==1 and ncols==1:self.axes=[self.fig.add_subplot(111)]
        else:self.axes=self.fig.subplots(nrows,ncols,sharex=False,sharey=False).flatten()
        self._sty(title);self.fig.tight_layout(pad=1.0)
    def _sty(self,title=''):
        for ax in self.axes:
            ax.set_facecolor(T.S1);ax.tick_params(colors=T.DIM,labelsize=6)
            for s in ax.spines.values():s.set_color(T.OVR);ax.grid(True,alpha=0.1,color=T.DIM)
        if title:self.axes[0].set_title(title,color=T.SUB,fontsize=9,pad=5)
    def clear_all(self):
        self.fig.clear();self.axes=[self.fig.add_subplot(111)];self._sty();self.fig.tight_layout(pad=1.0)

class LossPlot(PlotCanvas):
    def __init__(self):super().__init__('Training Loss');self.losses=[]
    def update_plot(self,ep,l):
        self.losses.append((ep,l));self.clear_all();self._sty('Training Loss')
        if self.losses:
            e,v=zip(*self.losses);self.axes[0].plot(e,v,color=T.PRI,lw=1.2);self.axes[0].fill_between(e,v,alpha=0.06,color=T.PRI)
        self.draw_idle()

class MetricPlot(PlotCanvas):
    def __init__(self):super().__init__('Metric');self.data=[]
    def update_plot(self,ep,m,nt=''):
        self.data.append((ep,m));self.clear_all()
        lb={'mlp':'Accuracy','cnn':'Accuracy','rnn':'R²','lstm':'R²','transformer':'R²','gan':'G Score'}.get(nt,'Metric')
        self._sty(lb)
        if self.data:
            e,v=zip(*self.data);c=T.GRN if nt in('mlp','cnn') else T.PEA
            self.axes[0].plot(e,v,color=c,lw=1.2);self.axes[0].fill_between(e,v,alpha=0.06,color=c)
        self.draw_idle()

class LRPlot(PlotCanvas):
    def __init__(self):super().__init__('Learning Rate');self.data=[]
    def update_plot(self,ep,lr):
        self.data.append((ep,lr));self.clear_all();self._sty('Learning Rate')
        if self.data:
            e,v=zip(*self.data);self.axes[0].plot(e,v,color=T.YEL,lw=1.2);self.axes[0].fill_between(e,v,alpha=0.08,color=T.YEL)
        self.draw_idle()

class GradFlowPlot(PlotCanvas):
    def __init__(self):super().__init__('Gradient Flow')
    def update_plot(self,gn):
        self.clear_all();self._sty('Gradient Flow (log ‖∇‖)')
        if not gn:self.draw_idle();return
        names=list(gn.keys());vals=list(gn.values())
        short=[n.split('.')[-1] if '.' in n else n for n in names]
        cols=[T.GRN if v>1e-4 else(T.YEL if v>1e-6 else T.RED) for v in vals]
        self.axes[0].barh(range(len(short)),vals,color=cols,alpha=0.8)
        self.axes[0].set_yticks(range(len(short)));self.axes[0].set_yticklabels(short,fontsize=5)
        self.axes[0].set_xscale('log');self.axes[0].axvline(x=1e-6,color=T.RED,ls='--',alpha=0.4,lw=0.7)
        self.fig.tight_layout(pad=1.0);self.draw_idle()

class OutputPlot(PlotCanvas):
    def __init__(self):super().__init__('Network Output')
    def update_plot(self,od):
        self.clear_all()
        if not od or od.get('type') is None:
            self.axes[0].text(0.5,0.5,'Waiting...',transform=self.axes[0].transAxes,ha='center',va='center',color=T.DIM)
            self.draw_idle();return
        t=od['type']
        if t=='boundary':
            self._sty('Decision Boundary');gx,gy,gp=od['gx'],od['gy'],od['gp'];r=int(math.sqrt(len(gx)))
            try:self.axes[0].contourf(gx.reshape(r,r),gy.reshape(r,r),gp.reshape(r,r),levels=20,alpha=0.55,cmap='RdYlBu')
            except:pass
            dx,dy,dc=od['dx'],od['dy'],od['dy2']
            self.axes[0].scatter(dx[dc==0],dy[dc==0],c=T.PRI,s=5,edgecolors='w',linewidth=0.2,label='C0')
            self.axes[0].scatter(dx[dc==1],dy[dc==1],c=T.PEA,s=5,edgecolors='w',linewidth=0.2,label='C1')
            if len(np.unique(dc))>2:self.axes[0].scatter(dx[dc==2],dy[dc==2],c=T.GRN,s=5,edgecolors='w',linewidth=0.2,label='C2')
            self.axes[0].legend(fontsize=5,facecolor=T.S1,edgecolor=T.OVR,labelcolor=T.TXT,markerscale=1.5)
        elif t=='seq':
            if 'attention' in od and od['attention'] and od['attention'][0] is not None:
                self.fig.clear();self.axes=[self.fig.add_subplot(121),self.fig.add_subplot(122)]
                for ax in self.axes:ax.set_facecolor(T.S1);ax.tick_params(colors=T.DIM,labelsize=6)
                for s in ax.spines.values():s.set_color(T.OVR);ax.grid(True,alpha=0.1,color=T.DIM)
                self.axes[0].set_title('Sequence Prediction',color=T.SUB,fontsize=9,pad=5)
                act,pred=od['act'],od['pred'];cols=[T.PRI,T.GRN,T.PEA]
                for i in range(min(3,act.shape[0])):
                    self.axes[0].plot(act[i,:,0],color=cols[i],lw=1.1,label=f'T{i+1}')
                    self.axes[0].plot(pred[i,:,0],color=cols[i],lw=1.1,ls='--',alpha=0.5,label=f'P{i+1}')
                self.axes[0].legend(fontsize=5,facecolor=T.S1,edgecolor=T.OVR,labelcolor=T.TXT,ncol=2)
                self.axes[1].set_title('Attention (Head 0, Layer 0)',color=T.SUB,fontsize=9,pad=5)
                aw=od['attention'][0]
                self.axes[1].imshow(aw,cmap='magma',aspect='auto',interpolation='nearest')
                self.axes[1].set_xlabel('Key',color=T.DIM,fontsize=6);self.axes[1].set_ylabel('Query',color=T.DIM,fontsize=6)
            else:
                self._sty('Sequence Prediction');act,pred=od['act'],od['pred'];cols=[T.PRI,T.GRN,T.PEA]
                for i in range(min(3,act.shape[0])):
                    self.axes[0].plot(act[i,:,0],color=cols[i],lw=1.1,label=f'True {i+1}')
                    self.axes[0].plot(pred[i,:,0],color=cols[i],lw=1.1,ls='--',alpha=0.5,label=f'Pred {i+1}')
                self.axes[0].legend(fontsize=5,facecolor=T.S1,edgecolor=T.OVR,labelcolor=T.TXT,ncol=2)
        elif t=='gan':
            self._sty('GAN: Real vs Generated');r,g=od['real'],od['gen']
            self.axes[0].scatter(r[:,0],r[:,1],c=T.PRI,s=4,alpha=0.5,label='Real',edgecolors='none')
            self.axes[0].scatter(g[:,0],g[:,1],c=T.RED,s=4,alpha=0.5,label='Gen',edgecolors='none')
            self.axes[0].set_xlim(-2.5,2.5);self.axes[0].set_ylim(-2.5,2.5);self.axes[0].set_aspect('equal')
            self.axes[0].legend(fontsize=5,facecolor=T.S1,edgecolor=T.OVR,labelcolor=T.TXT)
        self.fig.tight_layout(pad=1.0);self.draw_idle()

class ConfPlot(PlotCanvas):
    def __init__(self):super().__init__('Confusion Matrix')
    def update_plot(self,cd):
        self.clear_all();self._sty('Confusion Matrix')
        if not cd:self.draw_idle();return
        cm=conf_matrix(cd['true'],cd['pred'],cd['nc']);nc=cd['nc']
        self.axes[0].imshow(cm,cmap='Blues',interpolation='nearest')
        self.axes[0].set_xticks(range(nc));self.axes[0].set_yticks(range(nc))
        self.axes[0].set_xlabel('Pred',color=T.DIM,fontsize=6);self.axes[0].set_ylabel('True',color=T.DIM,fontsize=6)
        for i in range(nc):
            for j in range(nc):
                c='white' if cm[i,j]>cm.max()/2 else T.TXT
                self.axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',color=c,fontsize=7,fontweight='bold')
        self.fig.tight_layout(pad=1.0);self.draw_idle()

class FeatPlot(PlotCanvas):
    def __init__(self):super().__init__('Feature Space (PCA)')
    def update_plot(self,fd):
        self.clear_all();self._sty('Feature Space (PCA)')
        if not fd:self.draw_idle();return
        x,y,lb=fd['x'],fd['y'],fd['labels'];cols=[T.PRI,T.PEA,T.GRN,T.MAU,T.YEL,T.TEAL]
        for c in np.unique(lb):
            m=lb==c;self.axes[0].scatter(x[m],y[m],c=cols[int(c)%len(cols)],s=6,alpha=0.6,edgecolors='none',label=f'C{int(c)}')
        self.axes[0].legend(fontsize=5,facecolor=T.S1,edgecolor=T.OVR,labelcolor=T.TXT)
        self.fig.tight_layout(pad=1.0);self.draw_idle()

class WeightPlot(PlotCanvas):
    def __init__(self):super().__init__('Weight Distribution')
    def update_plot(self,wi):
        self.clear_all();self._sty('Weight Distribution')
        ms,ss=[],[]
        for k,v in wi.items():
            if k.endswith('_m') and not k.endswith('_gm'):ms.append(v)
            if k.endswith('_s') and not k.endswith('_gs'):ss.append(v)
        if ms:
            x=range(len(ms));self.axes[0].bar(x,ms,color=T.MAU,alpha=0.7,label='Mean')
            self.axes[0].errorbar(x,ms,yerr=ss,color=T.PEA,fmt='none',capsize=3,elinewidth=0.7)
            self.axes[0].set_xticks(list(x));self.axes[0].set_xticklabels([f'{i}' for i in x],fontsize=5)
        self.fig.tight_layout(pad=1.0);self.draw_idle()

# ═══════════════════════════════════════════════════════════════════════════════
#  DESCRIPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
DESCS={ 'mlp':"MLP: Fully connected layers with ReLU. Each neuron connects to all neurons in the next layer. Compare PyTorch nn.Linear vs TF Dense — same math, different APIs.",
        'cnn':"CNN: Conv1D filters slide over input detecting local patterns. Pooling reduces size. Compare PT Conv1d vs TF Conv1D — watch shared weights learn edges/waves.",
        'rnn':"RNN: Hidden state h_t carries temporal info: h_t=tanh(W_h·h_{t-1}+W_x·x_t). Limited by vanishing gradients. Compare PT nn.RNN vs TF SimpleRNN.",
        'lstm':"LSTM: Cell state C_t + gates (forget f, input i, candidate ĩ, output o). Solves vanishing gradient. Compare PT nn.LSTM vs TF LSTM layer.",
        'transformer':"Transformer: Self-attention Q,K,V → softmax(QK^T/√d)V. All positions in parallel. Compare PT MultiheadAttention vs TF MultiHeadAttention.",
        'gan':"GAN: G(z) generates fakes, D(x) scores real vs fake. Adversarial min-max game. Compare PT vs TF training loops — both use separate GradientTape/backprop."}
DATA_INFO={'mlp':"Binary classification on 2D data\nSelect dataset below",
           'cnn':"1D signal classification\n500×32, 3 wave classes",
           'rnn':"Next-step sine prediction\n300 seq×30 steps",
           'lstm':"Next-step sine prediction\n300 seq×30 steps",
           'transformer':"Next-step prediction via attention\n300 seq×30 steps",
           'gan':"Learn 2D distribution\nSelect target shape below"}

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle("🧠 NN Training Visualizer — PyTorch & TensorFlow")
        self.setMinimumSize(1320,840);self.resize(1480,900);self.setStyleSheet(T.css())
        self.ntype='mlp';self.fw='pytorch';self.thread=None;self.trained=False
        self._build_ui();self._on_type_changed('mlp')
        self.statusBar().showMessage("Ready")

    def _build_ui(self):
        cw=QWidget();self.setCentralWidget(cw);ml=QHBoxLayout(cw);ml.setContentsMargins(6,6,6,6);ml.setSpacing(6)

        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFixedWidth(278)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left=QWidget();ll=QVBoxLayout(left);ll.setContentsMargins(0,0,0,0);ll.setSpacing(4)

        # Framework
        g=QGroupBox("Deep Learning Framework");v=QVBoxLayout(g)
        self.fw_combo=QComboBox()
        self.fw_combo.addItem("🔥 PyTorch");self.fw_combo.setItemData(0,'pytorch')
        if TF_AVAILABLE:self.fw_combo.addItem("🧡 TensorFlow");self.fw_combo.setItemData(1,'tensorflow')
        self.fw_combo.currentIndexChanged.connect(self._on_fw_changed)
        v.addWidget(self.fw_combo)
        self.device_lbl=QLabel();self.device_lbl.setObjectName('desc')
        v.addWidget(self.device_lbl);ll.addWidget(g)

        # Network Type
        g=QGroupBox("Architecture");v=QVBoxLayout(g)
        self.type_combo=QComboBox()
        self.type_combo.addItems(['MLP','CNN','RNN','LSTM','Transformer','GAN'])
        self.type_combo.currentTextChanged.connect(lambda t:self._on_type_changed(t.lower()))
        v.addWidget(self.type_combo)
        self.desc_lbl=QLabel();self.desc_lbl.setObjectName('desc');self.desc_lbl.setWordWrap(True);self.desc_lbl.setMinimumHeight(44)
        v.addWidget(self.desc_lbl);ll.addWidget(g)

        # Optimizer
        g=QGroupBox("Optimizer");v=QFormLayout(g)
        self.opt_combo=QComboBox();self.opt_combo.addItems(['Adam','SGD','RMSprop','AdamW']);v.addRow("Type:",self.opt_combo)
        self.lr_spin=QDoubleSpinBox();self.lr_spin.setRange(0.0001,1.0);self.lr_spin.setValue(0.01);self.lr_spin.setSingleStep(0.001);self.lr_spin.setDecimals(4)
        v.addRow("Learning Rate:",self.lr_spin)
        self.wd_spin=QDoubleSpinBox();self.wd_spin.setRange(0,0.1);self.wd_spin.setValue(0);self.wd_spin.setSingleStep(0.001);self.wd_spin.setDecimals(4)
        v.addRow("Weight Decay:",self.wd_spin);ll.addWidget(g)

        # Regularization
        g=QGroupBox("Regularization");v=QVBoxLayout(g)
        self.dropout_cb=QCheckBox("Enable Dropout");v.addWidget(self.dropout_cb)
        h=QHBoxLayout();h.addWidget(QLabel("Rate:"))
        self.drop_spin=QDoubleSpinBox();self.drop_spin.setRange(0,0.8);self.drop_spin.setValue(0.2);self.drop_spin.setSingleStep(0.05)
        h.addWidget(self.drop_spin);v.addLayout(h);ll.addWidget(g)

        # Training Params
        g=QGroupBox("Training");v=QFormLayout(g)
        self.epoch_spin=QSpinBox();self.epoch_spin.setRange(10,5000);self.epoch_spin.setValue(200);self.epoch_spin.setSingleStep(50)
        v.addRow("Epochs:",self.epoch_spin)
        self.batch_spin=QSpinBox();self.batch_spin.setRange(4,256);self.batch_spin.setValue(32);self.batch_spin.setSingleStep(8)
        v.addRow("Batch Size:",self.batch_spin)
        self.seed_spin=QSpinBox();self.seed_spin.setRange(-1,99999);self.seed_spin.setValue(42)
        v.addRow("Seed:",self.seed_spin)
        h=QHBoxLayout();h.addWidget(QLabel("Speed:"))
        self.speed_slider=QSlider(Qt.Horizontal);self.speed_slider.setRange(0,100);self.speed_slider.setValue(5)
        h.addWidget(self.speed_slider);v.addLayout(h);ll.addWidget(g)

        # MLP Layers
        g=QGroupBox("MLP Layers");v=QVBoxLayout(g);self.mlp_group=g
        self.mlp_edit=QLineEdit("2, 64, 32, 2");v.addWidget(self.mlp_edit);ll.addWidget(g)

        # Dataset
        g=QGroupBox("Dataset");v=QVBoxLayout(g);self.data_group=g
        self.dataset_combo=QComboBox();self.dataset_combo.addItems(['Circles','Spirals','XOR','Moons'])
        v.addWidget(self.dataset_combo)
        self.data_lbl=QLabel();self.data_lbl.setObjectName('desc');self.data_lbl.setWordWrap(True)
        v.addWidget(self.data_lbl);ll.addWidget(g)

        # GAN
        g=QGroupBox("GAN Target");v=QVBoxLayout(g);self.gan_group=g
        self.gan_mode_combo=QComboBox();self.gan_mode_combo.addItems(['Ring','Spiral','Gaussian Mix'])
        v.addWidget(self.gan_mode_combo);ll.addWidget(g)

        # Controls
        g=QGroupBox("Controls");v=QVBoxLayout(g);v.setSpacing(4)
        r1=QHBoxLayout()
        self.start_btn=QPushButton("▶  Start");self.start_btn.setObjectName('startBtn');self.start_btn.clicked.connect(self._start)
        self.pause_btn=QPushButton("⏸  Pause");self.pause_btn.setObjectName('stopBtn');self.pause_btn.clicked.connect(self._pause);self.pause_btn.setEnabled(False)
        r1.addWidget(self.start_btn);r1.addWidget(self.pause_btn);v.addLayout(r1)
        r2=QHBoxLayout()
        self.step_btn=QPushButton("→  Step");self.step_btn.clicked.connect(self._step)
        self.reset_btn=QPushButton("↻  Reset");self.reset_btn.setObjectName('resetBtn');self.reset_btn.clicked.connect(self._reset)
        r2.addWidget(self.step_btn);r2.addWidget(self.reset_btn);v.addLayout(r2)
        r3=QHBoxLayout()
        self.save_btn=QPushButton("💾 Save");self.save_btn.clicked.connect(self._save_plots)
        self.export_btn=QPushButton("📋 CSV");self.export_btn.clicked.connect(self._export_csv)
        r3.addWidget(self.save_btn);r3.addWidget(self.export_btn);v.addLayout(r3)
        ll.addWidget(g)

        # Info
        g=QGroupBox("Info");v=QVBoxLayout(g)
        self.epoch_lbl=QLabel("Epoch: —");self.epoch_lbl.setStyleSheet("font-size:12px;font-weight:bold;color:"+T.PRI)
        self.loss_lbl=QLabel("Loss: —");self.loss_lbl.setStyleSheet("font-size:11px;color:"+T.PEA)
        self.metric_lbl=QLabel("Metric: —");self.metric_lbl.setStyleSheet("font-size:11px;color:"+T.GRN)
        self.lr_lbl=QLabel("LR: —");self.lr_lbl.setStyleSheet("font-size:11px;color:"+T.YEL)
        self.param_lbl=QLabel("Params: —");self.param_lbl.setStyleSheet("font-size:10px;color:"+T.DIM)
        self.time_lbl=QLabel("Time: —");self.time_lbl.setStyleSheet("font-size:10px;color:"+T.DIM)
        for w in[self.epoch_lbl,self.loss_lbl,self.metric_lbl,self.lr_lbl,self.param_lbl,self.time_lbl]:v.addWidget(w)
        ll.addWidget(g);ll.addStretch()
        scroll.setWidget(left);ml.addWidget(scroll)

        # Right
        right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.setSpacing(4)
        self.arch_widget=ArchWidget();rl.addWidget(self.arch_widget,stretch=2)

        self.tabs=QTabWidget()
        t1=QWidget();g1=QGridLayout(t1);g1.setContentsMargins(3,3,3,3);g1.setSpacing(3)
        self.loss_plot=LossPlot();self.metric_plot=MetricPlot();self.lr_plot=LRPlot();self.grad_plot=GradFlowPlot()
        g1.addWidget(self.loss_plot,0,0);g1.addWidget(self.metric_plot,0,1);g1.addWidget(self.lr_plot,1,0);g1.addWidget(self.grad_plot,1,1)
        self.tabs.addTab(t1,"📈 Training")

        t2=QWidget();g2=QGridLayout(t2);g2.setContentsMargins(3,3,3,3);g2.setSpacing(3)
        self.output_plot=OutputPlot();self.conf_plot=ConfPlot();self.feat_plot=FeatPlot();self.weight_plot=WeightPlot()
        g2.addWidget(self.output_plot,0,0,1,2);g2.addWidget(self.conf_plot,1,0);g2.addWidget(self.feat_plot,1,1);g2.addWidget(self.weight_plot,2,0,1,2)
        self.tabs.addTab(t2,"🔍 Analysis")

        t3=QWidget();v3=QVBoxLayout(t3);v3.setContentsMargins(3,3,3,3)
        self.log_table=QTableWidget();self.log_table.setColumnCount(5)
        self.log_table.setHorizontalHeaderLabels(['Epoch','Loss','Metric','LR','Avg‖∇‖'])
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers);self.log_table.setAlternatingRowColors(True)
        hh=self.log_table.horizontalHeader()
        for i in range(4):hh.setSectionResizeMode(i,QHeaderView.ResizeToContents)
        v3.addWidget(self.log_table);self.tabs.addTab(t3,"📊 Log")
        rl.addWidget(self.tabs,stretch=4);ml.addWidget(right,stretch=1)
        self._log_data=[];self._update_device_info()

    def _update_device_info(self):
        fw=self.fw
        if fw=='pytorch':
            dev='GPU ('+torch.cuda.get_device_name(0)+')' if torch.cuda.is_available() else 'CPU'
            self.device_lbl.setText(f"Device: {dev}\nCUDA: {'Yes' if torch.cuda.is_available() else 'No'}")
        elif TF_AVAILABLE:
            gpus=tf.config.list_physical_devices('GPU')
            self.device_lbl.setText(f"Device: {'GPU' if gpus else 'CPU'}\nGPUs: {len(gpus)}")
        else:
            self.device_lbl.setText("")

    def _on_fw_changed(self,idx):
        self.fw=self.fw_combo.itemData(idx) or 'pytorch'
        if self.thread and self.thread.isRunning():self._reset()
        self._on_type_changed(self.ntype);self._update_device_info()

    def _on_type_changed(self,nt):
        if self.thread and self.thread.isRunning():self._reset()
        self.ntype=nt;self.arch_widget.set_type(nt,self.fw,self._parse_mlp())
        self.desc_lbl.setText(DESCS.get(nt,''))
        self.mlp_group.setVisible(nt=='mlp');self.data_group.setVisible(nt in('mlp','cnn'))
        self.gan_group.setVisible(nt=='gan');self.dropout_cb.setVisible(nt in('mlp','cnn'))
        self.data_lbl.setText(DATA_INFO.get(nt,''))

    def _parse_mlp(self):
        try:
            ls=[int(x.strip()) for x in self.mlp_edit.text().split(',') if x.strip()]
            if len(ls)>=2 and all(x>0 for x in ls):return ls
        except:pass
        return[2,64,32,2]

    def _start(self):
        if self.trained:self._reset_plots()
        seed=self.seed_spin.value()
        if seed>=0:np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        if TF_AVAILABLE:tf.random.set_seed(seed) if seed>=0 else None
        self.start_btn.setEnabled(False);self.pause_btn.setEnabled(True);self.step_btn.setEnabled(False)
        for w in[self.type_combo,self.dataset_combo,self.mlp_edit,self.gan_mode_combo,self.fw_combo]:w.setEnabled(False)
        self.thread=TrainThread();self.thread.speed=self.speed_slider.value();self._t0=QDateTime.currentDateTime()
        dr=self.drop_spin.value() if self.dropout_cb.isChecked() else 0.0
        model=make_model(self.ntype,self.fw,self._parse_mlp(),dr)
        if model is None:self.statusBar().showMessage("Model creation failed!");self._enable_controls();return
        kw={'ntype':self.ntype,'fw':self.fw,'model':model,'lr':self.lr_spin.value(),
            'epochs':self.epoch_spin.value(),'bs':self.batch_spin.value(),'wd':self.wd_spin.value()}
        if self.ntype=='gan':
            mode=self.gan_mode_combo.currentText().lower().replace(' ','_');gd=make_gan_data(mode=mode)
            kw['gd']=gd;G,D=model;self.param_lbl.setText(f"G:{count_params(G,self.fw)} D:{count_params(D,self.fw)}")
        elif self.ntype in('mlp','cnn'):
            ds=self.dataset_combo.currentText().lower()
            if self.ntype=='mlp':
                Xd,Yd={'circles':make_circles,'spirals':make_spirals,'xor':make_xor,'moons':make_moons}[ds]()
            else:Xd,Yd=make_signals()
            kw['Xd']=Xd;kw['Yd']=Yd;nc=int(Yd.max()+1)
            self.param_lbl.setText(f"Params: {count_params(model,self.fw)} | Classes: {nc}")
        else:
            Xd,Yd=make_sine();kw['Xd']=Xd;kw['Yd']=Yd
            self.param_lbl.setText(f"Params: {count_params(model,self.fw)}")
        self.thread.setup(**kw)
        self.thread.epoch_sig.connect(self._on_epoch);self.thread.finished_sig.connect(self._on_finished)
        self.thread.start();self.statusBar().showMessage(f"Training {self.ntype.upper()} [{self.fw}]...")

    def _pause(self):
        if not self.thread:return
        if self.pause_btn.text().startswith("⏸"):self.thread.pause();self.pause_btn.setText("▶ Resume")
        else:self.thread.resume();self.pause_btn.setText("⏸ Pause")

    def _step(self):
        if self.trained:self._reset_plots()
        seed=self.seed_spin.value()
        if seed>=0:np.random.seed(seed);torch.manual_seed(seed)
        if TF_AVAILABLE:tf.random.set_seed(seed) if seed>=0 else None
        self.start_btn.setEnabled(False);self.pause_btn.setEnabled(False)
        for w in[self.type_combo,self.dataset_combo,self.mlp_edit,self.gan_mode_combo,self.fw_combo]:w.setEnabled(False)
        dr=self.drop_spin.value() if self.dropout_cb.isChecked() else 0.0
        model=make_model(self.ntype,self.fw,self._parse_mlp(),dr)
        if model is None:self._enable_controls();return
        kw={'ntype':self.ntype,'fw':self.fw,'model':model,'lr':self.lr_spin.value(),
            'epochs':1,'bs':self.batch_spin.value(),'wd':self.wd_spin.value()}
        if self.ntype=='gan':mode=self.gan_mode_combo.currentText().lower().replace(' ','_');kw['gd']=make_gan_data(mode=mode)
        elif self.ntype in('mlp','cnn'):
            ds=self.dataset_combo.currentText().lower()
            Xd,Yd={'circles':make_circles,'spirals':make_spirals,'xor':make_xor,'moons':make_moons}[ds]() if self.ntype=='mlp' else make_signals()
            kw['Xd']=Xd;kw['Yd']=Yd
        else:Xd,Yd=make_sine();kw['Xd']=Xd;kw['Yd']=Yd
        self.thread=TrainThread();self.thread.speed=self.speed_slider.value();self.thread.setup(**kw)
        self.thread.epoch_sig.connect(self._on_epoch);self.thread.finished_sig.connect(self._on_step_done)
        self.thread.start()

    def _on_step_done(self):self._enable_controls();self.trained=True;self.statusBar().showMessage("Step done")

    def _reset(self):
        if self.thread and self.thread.isRunning():self.thread.stop();self.thread.wait(2000)
        self._reset_plots();self._log_data=[];self.log_table.setRowCount(0);self.trained=False
        self._enable_controls()
        for l,t in[(self.epoch_lbl,"Epoch: —"),(self.loss_lbl,"Loss: —"),(self.metric_lbl,"Metric: —"),
                    (self.lr_lbl,"LR: —"),(self.param_lbl,"Params: —"),(self.time_lbl,"Time: —")]:l.setText(t)
        self.pause_btn.setText("⏸ Pause");self.statusBar().showMessage("Reset")

    def _enable_controls(self):
        self.start_btn.setEnabled(True);self.pause_btn.setEnabled(False);self.step_btn.setEnabled(True)
        for w in[self.type_combo,self.dataset_combo,self.mlp_edit,self.gan_mode_combo,self.fw_combo]:w.setEnabled(True)

    def _reset_plots(self):
        for p in[self.loss_plot,self.metric_plot,self.lr_plot,self.grad_plot,self.output_plot,self.conf_plot,self.feat_plot,self.weight_plot]:
            p.clear_all();p.draw_idle()
        self.loss_plot.losses=[];self.metric_plot.data=[];self.lr_plot.data=[]

    def _on_epoch(self,ep,loss,metric,wi,od):
        lr=wi.get('lr',self.lr_spin.value());gn=wi.get('grad_norms',{});avg_gn=np.mean(list(gn.values())) if gn else 0
        self.epoch_lbl.setText(f"Epoch: {ep}/{self.epoch_spin.value()}")
        self.loss_lbl.setText(f"Loss: {loss:.5f}");self.lr_lbl.setText(f"LR: {lr:.6f}")
        if self.ntype in('mlp','cnn'):self.metric_lbl.setText(f"Accuracy: {metric:.2%}")
        elif self.ntype=='gan':self.metric_lbl.setText(f"G Score: {metric:.4f}")
        else:self.metric_lbl.setText(f"R²: {metric:.4f}")
        dt=QDateTime.currentDateTime().msecsTo(self._t0);self.time_lbl.setText(f"Time: {abs(dt)/1000:.1f}s")
        self.loss_plot.update_plot(ep,loss);self.metric_plot.update_plot(ep,metric,self.ntype)
        self.lr_plot.update_plot(ep,lr);self.grad_plot.update_plot(gn);self.weight_plot.update_plot(wi)
        if od and od.get('type'):
            self.output_plot.update_plot(od)
            if 'confusion' in od:self.conf_plot.update_plot(od['confusion'])
            if 'features' in od:self.feat_plot.update_plot(od['features'])
        ms=f"{metric:.2%}" if self.ntype in('mlp','cnn') else f"{metric:.4f}"
        self._log_data.append([ep,f"{loss:.5f}",ms,f"{lr:.6f}",f"{avg_gn:.6f}"])
        if len(self._log_data)>500:self._log_data=self._log_data[-500:]
        row=self.log_table.rowCount();self.log_table.insertRow(row)
        for c,v in enumerate(self._log_data[-1]):
            it=QTableWidgetItem(str(v));it.setTextAlignment(Qt.AlignCenter);self.log_table.setItem(row,c,it)
        self.log_table.scrollToBottom();self.thread.speed=self.speed_slider.value()
        self.statusBar().showMessage(f"[{self.fw}] Epoch {ep} | Loss: {loss:.5f} | {ms}")

    def _on_finished(self):
        self._enable_controls();self.trained=True;self.pause_btn.setText("⏸ Pause")
        self.statusBar().showMessage(f"Training complete ✓ [{self.fw}]")

    def _save_plots(self):
        path,_=QFileDialog.getSaveFileName(self,"Save","nn_training.png","PNG (*.png)")
        if not path:return
        for name,p in[('loss',self.loss_plot),('metric',self.metric_plot),('lr',self.lr_plot),
                       ('grad',self.grad_plot),('output',self.output_plot),('conf',self.conf_plot),
                       ('feat',self.feat_plot),('weight',self.weight_plot)]:
            p.fig.savefig(path.replace('.png',f'_{name}.png'),dpi=150,bbox_inches='tight',facecolor=T.S1)
        self.statusBar().showMessage(f"Saved to {path.replace('.png','_*.png')}")

    def _export_csv(self):
        if not self._log_data:self.statusBar().showMessage("No data");return
        path,_=QFileDialog.getSaveFileName(self,"Export","log.csv","CSV (*.csv)")
        if not path:return
        with open(path,'w',newline='') as f:
            w=csv.writer(f);w.writerow(['Epoch','Loss','Metric','LR','Avg_Grad'])
            w.writerows(self._log_data)
        self.statusBar().showMessage(f"Exported to {path}")

    def closeEvent(self,e):
        if self.thread and self.thread.isRunning():self.thread.stop();self.thread.wait(3000)
        e.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);app.setStyle('Fusion')
    pal=app.palette()
    pal.setColor(QPalette.ColorRole.Window,T.qc(T.BG));pal.setColor(QPalette.ColorRole.WindowText,T.qc(T.TXT))
    pal.setColor(QPalette.ColorRole.Base,T.qc(T.S1));pal.setColor(QPalette.ColorRole.Text,T.qc(T.TXT))
    pal.setColor(QPalette.ColorRole.Button,T.qc(T.S2));pal.setColor(QPalette.ColorRole.ButtonText,T.qc(T.TXT))
    pal.setColor(QPalette.ColorRole.Highlight,T.qc(T.PRI));pal.setColor(QPalette.ColorRole.HighlightedText,T.qc(T.BG))
    pal.setColor(QPalette.ColorRole.AlternateBase,T.qc(T.S2));app.setPalette(pal)
    win=MainWindow();win.show();sys.exit(app.exec())