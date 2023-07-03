import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,random_split
import torchaudio
# import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import csv
import torchaudio

from sklearn.model_selection import train_test_split


def string_convert(s):
    s = s[1:-1]
    double_list = [float(s) for s in s.split(',')]
    # print(type(double_list))
    # print(len(double_list))
    # print(double_list[1])
    return double_list


class Dataload(Dataset):
    def __init__(self, user_name):
        # load landmark data.
        self.data = []
        csv_filename = 'data/{}/{}_2D_data.csv'.format(user_name, user_name)
        # 从 CSV 文件中读取数据和标签
        df_read = pd.read_csv(csv_filename)
        read_label_list = df_read['label'].tolist()
        read_data_list = df_read['data'].apply(eval).tolist()
        # 转换读取的数据为 Torch 张量
        read_data = torch.tensor(read_data_list)
        read_label = torch.tensor(read_label_list)  # 假设只有一个标签
        label_count = [0] * 7
        for data, label in zip(read_data, read_label):
            #print("data shape {} and label shape {}".format(data.shape, label))
            final_data = torch.squeeze(data)
            self.data.append((final_data, label))
            label_count[int(label)] += 1
        print("data count {}".format(label_count))
        print(read_label)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, hidden):
        x = x.float()
        
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


def train(model, epoch):
    print("training---------------")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(str(data) +  " " + str(target))
        data = data.to(device)
        target = target.to(device)
        # print("data size: " + str(data.size()))
        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))


def test(model, epoch):
    model.eval()
    correct = 0
    y_pred, y_target = [], []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        
        pred = torch.max(output, dim=1).indices
        print("the target is {} and the pred is {}".format(target, pred))
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# pd_load()


hyperparameters = {"lr": 0.005, "weight_decay": 0.0001, "batch_size": 1, "in_feature": 1064, "out_feature": 7}

device = torch.device("cpu")

user_name = input("please enter your name: ")

# train_set = Dataload('C:\study\Master study\\research\projectfile\expressionProject\data\\2D_time_data_xie.csv')
train_set = Dataload(user_name=user_name)


kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_ration = 0.8
test_ration = 1-train_ration

num_samples = len(train_set)
train_size = int(train_ration * num_samples)
test_size = num_samples - train_size

train_dataset, test_dataset = random_split(train_set, [train_size, test_size])
print("Training set size: " + str(len(train_dataset)))
print("Test set size: " + str(len(test_dataset)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
print("Train_loader set size: " + str(len(train_loader)))
# print("Test_loader set size: " + str(len(test_loader)))

model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping

log_interval = 10
for epoch in range(1, 50):
    # scheduler.step()
    train(model, epoch)
    test(model, epoch)
torch.save(model,'models/{}'.format(user_name))


# # load model
# load_model = torch.load('models/{}'.format(user_name))

# for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         print(data.shape)
#         output, hidden_state = load_model(data, model.init_hidden(hyperparameters["batch_size"]))
#         pred = torch.max(output, dim=1).indices
#         print(pred)
