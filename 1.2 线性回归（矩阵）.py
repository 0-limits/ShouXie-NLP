
# 注：没有使用dataset

import numpy as np
import pandas as pd


def get_data(file = "上海二手房价.csv"):
    datas = pd.read_csv(file, names = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], skiprows = 1)

    y = datas['y'].values.reshape(-1,1)
    X = datas[[f'x{i}' for i in range(1,7)]].values

    # z-score 归一化： (x - mean_x) / std_
    mean_y = np.mean(y)
    std_y = np.std(y)

    mean_X = np.mean(X, axis = 0, keepdims = True)
    std_X = np.std(X, axis = 0, keepdims = True)

    y = (y - mean_y) / std_y
    X = (X - mean_X) / std_X

    return X, y, mean_y, std_y, mean_X, std_X


if __name__ == '__main__':

    X, y, mean_y, std_y, mean_X, std_X = get_data()
    K = np.random.random((6,1))
    b = 0

    lr = 0.1
    epoch = 100

    for e in range(epoch):
        pre = X @ K + b
        loss = np.mean((pre - y)**2)        # loss为了打印所以取平均

        # G = 2 * (pre - y)       # G = dloss/dpre 上游导数
        G = (pre - y) / len(y)      # 上面这样梯度比较大，loss会nan，需要lr调到0.001才可以，所以这里弄小一点
        dk = X.T @ G
        db = np.mean(G)
        # db = np.sum(G)              # 视频没细说sum和mean，实测效果G要是求平均,b也求平均;G要是sum,b也sum,这样收敛快一点

        K = K - dk * lr
        b = b - db * lr

        print(f'loss = {loss:.3f}')


    while True:
        bedroom = (int(input('几室：')))
        ting = (int(input('几厅：')))
        wei = (int(input('几卫：')))
        area = (int(input('多少平：')))
        floor = (int(input('几楼：')))
        year = (int(input('你要预测哪一年的房价：')))

        test_x = (np.array([bedroom, ting, wei ,area, floor, year]).reshape(1,-1) - mean_X) / std_X

        predict = test_x @ K + b
        predict = predict * std_y + mean_y
        print(f"预测房价为：{predict}")

