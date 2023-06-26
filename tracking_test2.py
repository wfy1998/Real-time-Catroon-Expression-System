import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
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

from argparse import ArgumentParser

import cv2

import queue

import sounddevice as sd

import threading
# init labels
EMOTIONS = ["natural","expression_1","expression2","expression3","expression4","expression5","expression6"]


# 设置音频采样率和每个数据块的持续时间
sample_rate = 16000  # 采样率
chunk_duration = 1 # 持续时间（以秒为单位）
chunk_samples = int(sample_rate * chunk_duration)  # 每个数据块的样本数
hop_length = sample_rate //28
# 初始化 MFCC 转换器
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=12)
mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length-2)

# 创建一个队列来存储音频数据
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    # 将音频数据放入队列
    audio_queue.put(indata.copy())

def stream_thread_function(audio_callback, sample_rate, chunk_samples):
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=chunk_samples):
        while True:
            pass  # Or do some processing related to the stream, if required

def process_audio():
    print("process audio")
    # 从队列中取出音频数据
    indata = audio_queue.get()
    print("getting data")
    # 音频数据从 numpy 数组转换为 PyTorch 张量
    audio_tensor = torch.from_numpy(indata).float().to(torch.float32)
    print(audio_tensor)

    soundData = torch.mean(audio_tensor, dim=0, keepdim=True)
    tempData = torch.zeros([1, 16000])  # tempData accounts for audio clips that are too short
    if soundData.numel() < 16000:
        tempData[:, :soundData.numel()] = soundData
    else:
        tempData = soundData[:, :16000]

    soundData = tempData

    # 计算 MFCC
    mel_specgram = mel(soundData)
    mel_specgram = mel_specgram[:,:,:28]
    mel_specgram = torch.squeeze(mel_specgram)

    # 在这里处理或输出 MFCC
    print("mfcc: ")
    print(mel_specgram.shape)


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

def main():
    stream_thread = threading.Thread(target=stream_thread_function, args=(audio_callback, sample_rate, chunk_samples))
    stream_thread.start()
    # load model
    load_model = torch.load('models/{}'.format('wfy'))
    device = torch.device("cpu")
    cap = cv2.VideoCapture(args.cam)
    print(load_model)
    success, img = cap.read()
    while cap.isOpened():
        success, img = cap.read()
        cv2.imshow('Facial landmark', img)
        
        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_normalization", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=True)

    parser.add_argument("--port", type=int, 
                        help="specify the port of the connection to unity. Have to be the same as in Unity", 
                        default=5066)

    parser.add_argument("--cam", type=int,
                        help="specify the camera number if you have multiple cameras",
                        default=0)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)
    
    parser.add_argument("--feedback", action="store_true",
                        help="collect wrong result and store in dataset",
                        default=False)
    args = parser.parse_args()

    # demo code
    main()