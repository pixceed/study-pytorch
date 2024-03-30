
import pandas as pd


# データ読み込み
df = pd.read_csv('input/train_master.tsv', delimiter='\t')

print(df.head())
print()

# 入力変数と目的変数の切り分け
x = df.drop('expression', axis=1)
t = df['expression']

print(x.head(3))
print()

print(x.shape, t.shape)
print()