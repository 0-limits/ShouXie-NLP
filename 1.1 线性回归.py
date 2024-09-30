from random import shuffle

import numpy as np
from matplotlib.pyplot import yscale


class Dataset:
    def __init__(self, xs1, xs2, ys, shuffle, batch_size):
        self.xs1 = xs1
        self.xs2 = xs2
        self.ys = ys
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        return Dataloader(self)

    def __len__(self):
        return len(self.xs1)

class Dataloader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0

        self.indexes = np.arange(len(self.dataset))

        if self.dataset.shuffle:
            np.random.shuffle(self.indexes)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        index = self.indexes[self.cursor : self.cursor + self.dataset.batch_size]

        x1 = self.dataset.xs1[index]
        x2 = self.dataset.xs2[index]
        y = self.dataset.ys[index]

        self.cursor += self.dataset.batch_size

        return x1, x2, y


if __name__ == '__main__':
    years = np.array([i for i in range(2000,2025)])
    years = (years - 2000) / 22
    prices = np.array([10000,11000,12000,13000,14000, 12000,13000,16000,18000,19000,
                       22000,24000,23000,26000,25000, 29000,30000,26000,31000,31000,
                       32000,33000,33000,34000,35000]) / 35000
    floors = np.array([i for i in range(25,0,-1)])
    floors = floors / 25

    k1 = 1
    k2 = 0
    b = 1
    lr = 0.0001
    epoch = 10000

    batch_size = 2
    shuffle = True

    dataset = Dataset(years, floors, prices, shuffle, batch_size)

    for e in range(epoch):
        loss_sum = 0
        # for x,label in zip(years,prices):
        for year, floor, price in dataset:
            predict = k1 * year + k2 * floor + b
            loss = (predict - price) ** 2
            loss_sum += loss

            dk1 = 2 * (k1 * year + k2 * floor + b - price) * year
            dk2 = 2 * (k1 * year + k2 * floor + b - price) * floor
            db = 2 * (k1 * year + k2 * floor + b - price)

            k1 = k1 - dk1 * lr
            k2 = k2 - dk2 * lr
            b = b - db * lr
        # print(f"[{e}]:","loss = ", loss_sum / len(dataset))
        # print(f"[{e}]:",'k2 = ', k2)
        print(f"[{e}]:",'k1 = ', k1)

    print(f"k1 = {k1}, k2 = {k2}, b = {b}")


    for f in range(1,26):
        year = ( 2024 - 2000 ) / 22
        floor = f
        print(f"{f}楼 预测房价：", (k1 * year + k2 * floor + b) * 35000)

