import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.nn.functional import one_hot, softmax


# 从MNIST原文件加载到np array
def load_labels(file):
    with open(file, 'rb') as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype = np.int32)

def load_images(file):
    with open(file, 'rb') as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack('>iiii', data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype = np.uint8).reshape(num_items, -1)

# 把MNIST原标签转换成onehot形式
def make_onehot(labels, class_num = 10):
    result = np.zeros((len(labels),class_num))
    for index, label in enumerate(labels):
        result[index][label] = 1
    return result

def sigmoid(x):
   return 1/(1+np.exp(-x))

def softmax(x):
    ex = np.exp(x)
    ex_sum = np.sum(ex, axis = 1, keepdims = True)
    return ex/ex_sum

if __name__ == '__main__':
    train_datas = load_images('train-images-idx3-ubyte') / 255          # 归一化
    train_label = make_onehot(load_labels('train-labels-idx1-ubyte'))

    test_datas = load_images('t10k-images-idx3-ubyte') / 255
    test_label = load_labels('t10k-labels-idx1-ubyte')          # 测试集不用onehot，更好比较acc

    epoch = 50
    batch_size = 16        # batch_size = 6000 只能收敛到 acc=46%，=200 收敛到 acc=84%；=100 时 acc=87%；=50 时 acc=89%；=24 时 acc=93%
                           # 可能是因为batch size小，更新次数更多次。（epoch=10时的测试数据）      # acc最高95.5%
    lr = 0.1

    hidden_num = 256        # 隐藏层数量可以改
    W1 = np.random.normal(0,1,size = (784,hidden_num))      # 正态分布，不是0~1的分布
    W2 = np.random.normal(0,1,size = (hidden_num,10))

    batch_times = int(np.ceil(len(train_datas) / batch_size) )    # np.ceil 向上取整，避免无法整除

    for e in range(epoch):
        loss_mean = 0
        for batch_index in range(batch_times):
            batch_x = train_datas[ batch_index*batch_size : (batch_index+1)*batch_size ]
            batch_label = train_label[ batch_index*batch_size : (batch_index+1)*batch_size ]

            # forward
            h = batch_x @ W1
            sig_h = sigmoid(h)
            predict = sig_h @ W2
            predict = softmax(predict)

            loss = - np.sum(batch_label * np.log(predict))/batch_size      # 因为log（0~1）是负数，所以加个负号，行业习惯
            loss_mean += - np.sum(batch_label * np.log(predict))

            # backward 求导
            G = (predict - batch_label) / batch_size                       # 除以batch_size，不然batch_size=6000时，会nan。loss也除以batch_size，这样batch_size改变时loss就不会变大
                                                                           # 其实这是常规操作，视频讲的不好    # G的求导背下来就可以
            dW2 = sig_h.T @ G
            dsig_h = G @ W2.T
            dh = dsig_h * (sig_h * (1-sig_h))
            dW1 = batch_x.T @ dh

            W1 = W1 - dW1 * lr
            W2 = W2 - dW2 * lr

            # print(f'第{e}轮 第{batch_index} batch：', loss)

        # print(loss)
        loss_mean = loss_mean / len(train_datas)
        # print(loss_mean)

        # test 获得accuracy准确率
        h = test_datas @ W1
        sig_h = sigmoid(h)
        predict = sig_h @ W2
        predict = softmax(predict)

        predict = np.argmax(predict,axis = 1)
        acc = np.mean(predict == test_label)

        print(f'acc = {acc}, loss_mean = {loss_mean}')

    # 可视化查看预测结果（测试集）
    while True:
        index = int(input('输入测试集图片序号（0~9999）：'))
        print(f'预测数字：{predict[index]}, 标签数字：{test_label[index]}')
        plt.imshow(test_datas[index].reshape(28,28))
        plt.show()


    # 可视化一张图片（训练集）
    # img = train_datas[10086]
    # plt.imshow(img.reshape(28,28))
    # plt.show()

