import numpy as np
import sounddevice as sd
import torch
import torchaudio
import queue
import threading
import torch
import torchaudio
"""
Main program to run the detection and TCP
"""

from argparse import ArgumentParser
from cgi import test
from statistics import quantiles
import cv2
import mediapipe as mp
import numpy as np
from collections import deque 
import pyrr
import keyboard
import csv

# for TCP connection with unity
import socket

# face detection and facial landmark
from get_facial_landmark import FaceMeshDetector

# pose estimation and stablization
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# Miscellaneous detections (eyes/ mouth...)
from get_facial_features import FacialFeatures, Eyes

import sys

# tensorflow
import tensorflow as tf

import numpy as np
import sounddevice as sd
import torch
import torchaudio
import queue
import threading

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
duration = 2



def audio_record():
    global audio_frames
    while(1):
        audio_frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        print(audio_frames.shape)

# def stream_thread_function(audio_callback, sample_rate, chunk_samples):
#     with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=chunk_samples):
#         while True:
#             pass  # Or do some processing related to the stream, if required


def classification(data,reconstructed_model):
    label = 0
    if not data:
        return 0
    arr = np.array(data)
    if arr.shape[0] != 28:
        return 0
    # print(arr.shape)
    row = arr.reshape(1,28,936,1)
    # print(row.shape)
    result = reconstructed_model.predict(row)
    emotion_probability = np.max(result)
    label = EMOTIONS[result.argmax()]
    # print(result)s
    if(result[0][1] > 0.50):
        label = 1
        # print("expression_1")
    elif(result[0][2] > 0.90):
        label = 2
        # print("expression_2")
    elif(result[0][3] > 0.90):
        label = 3
    elif(result[0][4] > 0.90):
        label = 4
    elif(result[0][5] > 0.90):
        label = 5
    elif(result[0][6] > 0.90):
        label = 6    
    else:
        label = 0
        # print("expression_2")                     
        # print("expression_0")
    print(label)
    return label



def process_audio():
    print("process audio")
    # 从队列中取出音频数据
    # indata = audio_queue.get()
    indata = audio_frames
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
    return mel_specgram

# # 创建输入流
# stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=chunk_samples)


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


def data_combination(time_data, audio_data):
    numpy_d = np.array(time_data)
    tem = torch.from_numpy(numpy_d)
    # print("tem")
    # print(tem.shape)
    # print(audio_data.shape)
    final_data = torch.cat((tem.T, audio_data),dim=0) # correct 28, x(time, feature)
    final_data = final_data.T
    final_data = final_data.unsqueeze(0)
    return final_data # 1064

def main():

    
    # Classification data
    classification_result = 0
    time_data = deque()
    frame = 28
    frame_count = 0

    threading.Thread(target=audio_record).start()

    # model init
    load_model = torch.load('models/{}'.format('wfy'))
    device = torch.device("cpu")
    hyperparameters = {"lr": 0.005, "weight_decay": 0.0001, "batch_size": 1, "in_feature": 1064, "out_feature": 7}

    

    # use internal webcam/ USB camera
    cap = cv2.VideoCapture(args.cam)

    # IP cam (android only), with the app "IP Webcam"
    # url = 'http://192.168.0.102:4747/video'
    # url = 'https://192.168.0.102:8080/video'
    # cap = cv2.VideoCapture(url)

    # Facemesh
    detector = FaceMeshDetector()

    # get a sample frame for pose estimation img
    success, img = cap.read()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue
        img_facemesh, faces, face_row_3d,face_row_2d = detector.findFaceMesh(img)
        if len(face_row_2d) == 0:
            print("no data")
            continue

        # generate relative time serier data
        if frame_count < frame:
            # time_data.append(relative_face)
            time_data.append(face_row_2d)
            frame_count += 1
            print(frame_count)
        else:
            print("data full")
            time_data.popleft()
            # time_data.append(relative_face)
            time_data.append(face_row_2d)
            audio_data = process_audio()
            finaldata = data_combination(time_data, audio_data)
            print(finaldata.shape)
            # get label
            finaldata = finaldata.to(device)
            output, hidden_state = load_model(finaldata, load_model.init_hidden(hyperparameters["batch_size"]))
            pred = torch.max(output, dim=1).indices
            print(pred)
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
