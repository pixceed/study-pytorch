import numpy as np
import pandas as pd

import torch
import torch.nn as nn # ネットワークの構築
import torch.nn.functional as F # 様々な関数の使用
import torch.optim as optim # 最適化アルゴリズムの使用

import pytorch_lightning as pl
from pytorch_lightning import Trainer

# ＜ -- 乱数シードの固定 -- ＞
def setup_all_seed(seed=0):
    # numpyに関係する乱数シードの設定
    np.random.seed(seed)

    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

setup_all_seed()


# ＜ -- Wine データセットの準備 -- ＞
# データ読み込み
df = pd.read_csv('input/wine_class.csv')

# データの表示（先頭の5件）
# print(df.head())

# 入力変数と目的変数の切り分け
print(np.unique(df['Class'], return_counts=True))
x = df.drop('Class', axis=1)
t = df['Class']

print(x.head(3))
print()

print(x.shape, t.shape)
print()

# tensorに変換
x = torch.tensor(x.values, dtype=torch.float32)
t = torch.tensor(t.values, dtype=torch.int64) - 1 # ラベルを0~2にするため

# 入力変数と目的変数をまとめて、データセットを作成
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val: test = 60%　: 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int((len(dataset) - n_train) * 0.5)
n_test = len(dataset) - n_train - n_val

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

# ＜ -- モデル定義 -- ＞

# 学習データに対する処理
class TrainNet(pl.LightningModule):

    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results


# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'val_loss': loss, 'val_acc': acc}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        results = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results


# テストデータに対する処理
class TestNet(pl.LightningModule):

    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):

    def __init__(self, input_size=10, hidden_size=5, output_size=3, batch_size=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def lossfun(self, y, t):
        return F.cross_entropy(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

# ＜ -- 学習 -- ＞
model = Net()

trainer = Trainer()

trainer.fit(model)