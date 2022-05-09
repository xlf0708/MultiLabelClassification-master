'''
Descripttion: 
version: 
Author: ws
Date: 2021-07-09 17:47:55
LastEditors: ws
LastEditTime: 2021-08-28 16:17:23
'''
from models.network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import sys
from cal_xts import calculate_king_sys_suphx
from torch.utils.tensorboard import SummaryWriter

print(sys.path)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 2层卷积 1层全连接 输出
        # 输入数据shape(335, 27, 1)
        #
        #padding 需计算
        self.conv1 = nn.Conv2d(in_channels=335, out_channels=256, kernel_size=(3,1), padding=(1,0))
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=(3,1), padding=0)
        # in_features 计算
        self.fc1 = nn.Linear(in_features=32*25*1, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x=x.view(-1,32*25*1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.out(x), dim=1)
        # print(x.shape)

        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# net = Net().to(device)
net = ResNet(blocks=4,num_classes=4).to(device)
print(net)

# optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=net.parameters(), lr=0.0001)
#测试网络结构正确性


# json Dataset
class SRWTJSONDataset(Dataset):
    def __init__(self, path, phase):
        f = open(path, 'r', encoding='utf-8')
        self.json_data = f.readlines()  #
        self.idx_list = list(range(len(self.json_data)))
        # print(self.idx_list)
        # print("--------------",self.json_data["0"])
        random.shuffle(self.idx_list)
        self.phase = phase
    # 获取编码
    def __getitem__(self, index: int):
        data, label = calculate_king_sys_suphx(eval(self.json_data[self.idx_list[index]]))
        # print(label)
        return (data, label)

    def __len__(self):
        print(len(self.idx_list))
        return len(self.idx_list)

# hdf5 Dataset
# train_dataloader = DataLoader(dataset=ConcatDataset(make_data_list(18, 'train')), batch_size=512, num_workers=1)
# test_dataloader = DataLoader(dataset=ConcatDataset(make_data_list(2, 'val')), batch_size=512, num_workers=1)
# json Dataset
train_json_file = "../data/recommond_2022425.txt"
val_json_file ="../data/recommond_2022425.txt"

train_dataloader = DataLoader(dataset=SRWTJSONDataset(train_json_file,'train'), batch_size=256)
test_dataloader = DataLoader(dataset=SRWTJSONDataset(val_json_file,'val'), batch_size=256)

# train1
def train(train_dataloader, model, loss_fn, optimizer):
    size = len(train_dataloader.dataset)
    for batch, (x, y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)
        # print('标签shape',y.shape)
        # compute prediction error
        pred = model(x)
        # print("pred_shape",pred.shape)
        # print(y)
         # print(y.argmax(1))
        loss = loss_fn(pred, y.argmax(1))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar(tag="tran_loss",scalar_value=loss.item(),global_step= batch)
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f} ")

# test
def test(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    num_batch = len(test_dataloader)
    test_loss, acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y.argmax(1))
            test_loss += loss.item()
            acc += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batch
    acc /= size
    print(f"Test Error: \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# trainning
epochs = 100
writer = SummaryWriter(log_dir="logs")
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)
    test(test_dataloader, net, loss_fn)
print("Done!")

# json_file = "/media/ren2/My Passport/srmj_data/srwt_v0.json"
# file = open(json_file, 'r')
# json_data = json.load(file)
# print(json_data['1'])
writer.close()

