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
        csv_path = 'data/{}/2D_time_data_{}'.format(user_name, user_name)
        audio_folder = 'data/{}'.format(user_name)
        df = pd.read_csv('data/{}/2D_time_data_{}.csv'.format(user_name, user_name),header=None)
        audio_length = 16000
        hop_length = audio_length // 60
        X = df.iloc[: , 1:]
        y = df.iloc[: , :1]
        # print("X shape: " + str(X.shape))
        x = np.zeros((X.shape[0],X.shape[1],936))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x[i][j] = string_convert(X.values[i][j])
        # print("x shape: " + str(x[0].shape))
        # print("x val: " + str(x[0][0]))
        # audio index par
        audio_index = 0
        current_label = 0
        for i in range(X.shape[0]):
            transposed_data = x[i].T
            label = y.values[i][0]
            if label != current_label:
                current_label = label
                audio_index = 0
            # audio file
            audio_file = str(y.values[i][0])
            audio_file += '_{}.wav'.format(audio_index)
            audio_index += 1
            audio_path = os.path.join(audio_folder, audio_file)
            if os.path.isfile(audio_path) and audio_path.endswith(".wav"):
                # print("find the file")
                # print(audio_path)
                sound, sample_rate = torchaudio.load(audio_path)
            else:
                print("file {} doesn't exist".format(audio_path))

            soundData = torch.mean(sound, dim=0, keepdim=True)
            tempData = torch.zeros([1, 16000])  # tempData accounts for audio clips that are too short
            if soundData.numel() < 16000:
                tempData[:, :soundData.numel()] = soundData
            else:
                tempData = soundData[:, :16000]

            soundData = tempData
            
            mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length-2)(soundData)  # (channel, n_mels, time)
            mel_specgram = mel_specgram[:,:,:60]
            mel_specgram = torch.squeeze(mel_specgram)
            print("info")
            print(transposed_data.shape) # 936 60
            final_data = torch.cat((torch.from_numpy(transposed_data), mel_specgram),dim=0)
            print(final_data.shape) # 1064 60

            self.data.append((final_data.T, label)) 
            print(final_data.T.shape) # 60 1064
            print(type(label))
        # print(type(transposed_data[0][0]))
        # print(type(int(label)))

        # load audio data
        
 

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
        print("data shape is {}".format(data.shape))
        data = data.to(device)
        target = target.to(device)
        # print("data size: " + str(data.size()))
        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        # print("the out put is {}".format(output))
        # print("the target is {}".format(target))
        #the out put is tensor([[ 6.9740, -3.2779,  0.6113,  1.2845,  0.1642, -1.9407, -2.5017]],
        # grad_fn=<AddmmBackward0>)
        # the target is tensor([0])
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
print("Test set size: " + str(len(train_set)))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu

train_ration = 0.8
test_ration = 1-train_ration

num_samples = len(train_set)
train_size = int(train_ration * num_samples)
test_size = num_samples - train_size

train_dataset, test_dataset = random_split(train_set, [train_size, test_size])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
print("Train_loader set size: " + str(len(train_loader)))
# print("Test_loader set size: " + str(len(test_loader)))


model = torch.load('models/{}'.format(user_name))
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping

log_interval = 10
for epoch in range(1, 100):
    # scheduler.step()
    train(model, epoch)
    test(model, epoch)
# torch.save(model,'models/{}'.format(user_name))


# # load model
# load_model = torch.load('models/{}'.format(user_name))

# for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         print(data.shape)
#         output, hidden_state = load_model(data, model.init_hidden(hyperparameters["batch_size"]))
#         pred = torch.max(output, dim=1).indices
#         print(pred)
