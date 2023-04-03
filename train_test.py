import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
from tqdm import trange
from model import lstm
from ICUData import ICUData
def train_test_split(data, label, rate):
    if len(data) != len(label):
        return -1
    l = len(data)
    train_data = data[:round(l*rate)]
    test_data = data[round(l*rate)+1:]
    train_label = label[:round(l*rate)]
    test_label = label[round(l*rate)+1:]
    return train_data, train_label, test_data, test_label

def train(model,optimizer,epoch,dataloader):
    res = []
    size = len(dataloader.dataset)
    for i in trange(epoch):
        for batch,(X,y) in enumerate(dataloader):
            X=X.cuda()
            y=y.cuda()
            pred=model.forward(X)
            loss=model.loss(pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                res.append(loss)
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return res
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X=X.cuda()
            y=y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    learning_rate = 1e-3
    batch_size = 1
    epochs = 1
    model = lstm(20,25,batch_size)
    model.cuda()
    input_data = torch.load("input_data_averaged.pt")
    input_label = torch.load("input_label.pt")
    train_data, train_label, test_data, test_label = train_test_split(input_data, input_label, 0.8)
    trainSet = ICUData(train_data,train_label)
    testSet = ICUData(test_data,test_label)
    trainLoader = DataLoader(trainSet, batch_size=batch_size)
    testLoader = DataLoader(testSet, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    result = train(model,optimizer,epochs,trainLoader)
    test_loop(testLoader,model,model.loss)
    plt.plot(list(x for x in range(len(result))),result)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()