# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000  # 繰り返しの回数を適宜設定する
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


'''
実行結果
train acc, test acc | 0.0996166666667, 0.1035
train acc, test acc | 0.796033333333, 0.7997
train acc, test acc | 0.878116666667, 0.8839
train acc, test acc | 0.896983333333, 0.9008
train acc, test acc | 0.905833333333, 0.9087
train acc, test acc | 0.9137, 0.9153
train acc, test acc | 0.9183, 0.9212
train acc, test acc | 0.921633333333, 0.923
train acc, test acc | 0.92625, 0.9264
train acc, test acc | 0.930116666667, 0.9307
train acc, test acc | 0.931516666667, 0.9326
train acc, test acc | 0.934516666667, 0.9357
train acc, test acc | 0.9362, 0.9363
train acc, test acc | 0.93865, 0.9407
train acc, test acc | 0.941083333333, 0.9427
train acc, test acc | 0.943416666667, 0.9437
train acc, test acc | 0.944533333333, 0.9453
'''
