
# 注：没有使用dataset，没有使用归一化

import numpy as np

dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]], dtype=np.float32)  # 0
cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)       # 1

labels =  np.array([0]*7 + [1]*7, dtype=np.int32).reshape(-1,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == '__main__':
    X = np.vstack((dogs, cats))
    k = np.random.normal(0,1,size=(2,1))
    b = 0

    epoch = 1000
    lr = 0.1

    for e in range(epoch):
        predict = X @ k + b
        predict = sigmoid(predict)

        loss = np.mean(- labels * np.log(predict) + (1 - labels) * np.log(1 - predict) )     # log值是负数，所以加负号  # loss为了打印所以取平均

        G = predict - labels                # 这个记下来就行了 求导就别求了，数学不好 # dloss/dpredict

        dk = X.T @ G
        db = np.sum(G)
        # db = G

        k = k - dk * lr
        b = b - db * lr

        print(loss)


    while True:
        x1 = float(input('毛发长:'))
        x2 = float(input('腿长:'))
        test_X = np.array([x1,x2]).reshape(1,2)
        predict = sigmoid(test_X @ k + b)

        if predict > 0.5:
            print('类别是猫')
        elif predict <= 0.5:
            print('类别是狗')
        print(predict)