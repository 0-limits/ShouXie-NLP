import torchvision.datasets
import torch
from click.core import batch
from torch.utils import data
from torchvision import transforms
from torch import nn
import numpy as np

trans = transforms.ToTensor()

mnist_train = torchvision.datasets.MNIST(
    root = './MNIST_dataset', train = True, transform = trans, download=True
)
mnist_test = torchvision.datasets.MNIST(
    root = './MNIST_dataset', train = False, transform = trans, download=True
)

batch_size = 64
train_dataloader = data.DataLoader(mnist_train, batch_size=batch_size)
test_dataloader = data.DataLoader(mnist_test, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self, X):
        X = self.flatten(X)
        logit = self.linear_relu_stack(X)
        return logit


model  = NeuralNetwork().to(device = 0) # 查看device的代码？

model.load_state_dict(torch.load('model_MNIST.pth'))        # 从上一次训练的.pth加载
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

def train(dataloader, model, loss_fn, optimizer):

    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(0), y.to(0)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()   # 为什么这一句不能做到.step()里？
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss:{loss:>7f}  [{current:>5d}/ {len(dataloader.dataset):>5d}]')

def test(dataloader, model):

    model.eval()    # 这是什么？
    test_loss, correct = 0, 0
    with torch.no_grad():   # 这是什么with语句？
        for X,y in dataloader:
            X,y = X.to(0), y.to(0)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    #item()是什么？
            correct += (pred.argmax(1) == y).sum().item()      # item() 之前是tensor(5, device='cuda:0')，item 之后是 5
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()   不要type(torch.float)可以吗？可以
        test_loss /= len(dataloader.dataset)
        correct /= len(dataloader.dataset)
        print(f'Test Accuracy: {correct:>6f}, Test Loss: {test_loss:>8f}')

epochs = 1

for e in range(epochs):
    print(f'Epoch {e+1}----------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)

# 保存模型
torch.save(model.state_dict(), 'model_MNIST.pth')   # 为什么有些是.xxx()，有些是.xxx，怎么记得住，有什么规律，我怎么知道它怎么写，官网文档怎么看，一下子看不完
print('Saved Pytorch Model State to model_MNIST.pth .')

# 加载模型
model = NeuralNetwork()
model.load_state_dict(torch.load('model_MNIST.pth'))

print('')
#18

