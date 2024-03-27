import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ＜ -- ミニバッチ学習の準備 -- ＞
batch_size = 10

# DataLoader：datasetからバッチごとに取り出すことを目的に使用
train_loader = torch.utils.data.DataLoader(dataset=train,
                                        batch_size=batch_size,
                                        shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val,
                                        batch_size=batch_size,
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                        batch_size=batch_size,
                                        shuffle=True)


# ＜ -- モデル定義 -- ＞
class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(input_size)

    # 順伝播
    def forward(self, x):
        x = self.bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

input_size = 10
hidden_size = 5
output_size = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_size, hidden_size, output_size).to(device)
print(model)


# ＜ -- 学習の準備 -- ＞

# 目的関数(損失関数)の設定
criterion = nn.CrossEntropyLoss()

# 最適化法の指定　optimizer：最適化
# SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 1epochの訓練を行う関数の定義
def train_model(model, train_loader, criterion, optimizer, device='cpu'):
    train_loss = 0.0
    num_train = 0

    # 学習モデルに変換
    model.train()

    for i, (x, t) in enumerate(train_loader):

        # batch数をカウント
        num_train += len(t)

        # 学習時に使用するデバイスへデータの転送
        x = x.to(device)
        t = t.to(device)

        # パラメータの勾配を初期化
        optimizer.zero_grad()

        # 予測値の算出: 推論(順伝播)
        y = model(x)

        # 目標値と予測値から目的関数の値を算出: 損失の算出
        loss = criterion(y, t)

        # 各パラメータの勾配を算出: 誤差逆伝播
        loss.backward()

        # 勾配の情報を用いたパラメータの更新
        optimizer.step()

        # lossを加算
        train_loss += loss.item()

    # lossの平均値を取る
    train_loss = train_loss / num_train

    return train_loss

# 検証データによるモデル評価を行う関数の定義
def test_model(model, test_loader, criterion, optimizer, device='cpu'):

    test_loss = 0.0
    num_test = 0

    # modelを評価モードに変更
    model.eval()

    with torch.no_grad(): # 勾配計算の無効化
        for i, (x, t) in enumerate(test_loader):

            num_test += len(t)

            x = x.to(device)
            t = t.to(device)

            y = model(x)

            loss = criterion(y, t)
            test_loss += loss.item()

        # lossの平均値を取る
        test_loss = test_loss / num_test
    return test_loss

# ＜ -- 学習 -- ＞

num_epochs = 1000
train_loss_list = []
test_loss_list = []

for epoch in range(1, num_epochs+1, 1):

    train_loss = train_model(model, train_loader, criterion, optimizer, device=device)
    test_loss = test_model(model, test_loader, criterion, optimizer, device=device)

    print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)


# ＜ -- 学習済みモデルの保存 -- ＞
torch.save(model, 'output/iris_origin.pth')

# ＜ -- 学習推移のグラフ化 -- ＞
plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()


print()

# modelを評価モードに変更
model.eval()

with torch.no_grad(): # 勾配計算の無効化
    for i, (x, t) in enumerate(val_loader):

        x = x.to(device)
        t = t.to(device)

        y = model(x)
        y_label = torch.argmax(y, dim=1)

        acc = torch.sum(y_label == t) * 1.0 / len(t)

        print("##########################")
        # print(x)
        print(t)
        # print(y)
        print(y_label)
        print('Accuracy: {:.1f}%'.format(acc * 100))
        print()