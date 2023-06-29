import cv2
import numpy as np
import threading
import wave
import os
import mediapipe as mp
import time
import csv
import sounddevice as sd
import sys
import soundfile as sf
from pynput import keyboard
from scipy.io.wavfile import write
from argparse import ArgumentParser
from core.videosource import WebcamSource
from core.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)
# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,random_split
import torchaudio
import torch.optim as optim
import traceback
import queue
# mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)



# data normalization
landmark_number = 468 # total landmark points
points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# uncomment next line to use all points for PnP algorithm
points_idx = list(range(0,landmark_number)); points_idx[0:2] = points_idx[0:2:-1]
points_idx = list(range(0,landmark_number))
frame_height, frame_width, channels = (720, 1280, 3)
# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)
dist_coeff = np.zeros((4, 1))


# par setting
expressionID = 0 # expression ID
user_name = '' # user name
file_name_3D = '' # file name that store 2D data
file_name_3D = '' # file name that store 3D data
frame_step = 60 # vedio record length. default is 60 frame.
floder_name = '' # floder that store user's data
space_press = False # check if the space is pressed
audio_ID = 0
# set audio record
fs = 16000
seconds = 1

# 设置音频参数
chunk = 1024  # 记录的块大小
channels = 1  # 双声道
fs = 16000  # 记录频率

audio_data = []  # 用于储存音频帧的列表
recording_audio = False
recording_video = False
audio_queue = queue.Queue()
mel = torchaudio.transforms.MelSpectrogram(sample_rate=fs, hop_length=fs-2)

def on_press(key):
    global recording_audio,recording_video
    if key == keyboard.Key.space:  # 检测空格键
        print("change on press")
        recording_audio = recording_video = True
    elif key == keyboard.KeyCode.from_char('q'):
        # 当按下 'q' 键，停止监听器
        sys.exit()
        return False
    
    
def record_audio(duration = 1):
    global audio_frames,audio_ID, recording_audio,recording_video, user_name,expressionID
    print("开始录音...")
    while(1):
        recording_audio = False
        audio_frames = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        audio_queue.put(audio_data)
        #write('data/{}/{}_{}.wav'.format(user_name,str(expressionID), str(audio_ID)),  fs, audio_frames)  # 保存为WAV文件
        audio_ID +=1
        print("录音结束，保存中...ID is: {} size is {}".format(expressionID, audio_frames.shape))
    # print(audio_frames)
    # print(audio_frames.shape)

    
def process_audio():
    # print("process audio")
    # 从队列中取出音频数据
    indata = audio_queue.get()
    # indata = audio_frames
    print(audio_queue.queue)
    print("audio_frames shape in process audio is: {}".format(len(indata))) #记录会出现nan值
    # print("getting data")
    # 音频数据从 numpy 数组转换为 PyTorch 张量
    # audio_tensor = torch.from_numpy(indata).float().to(torch.float32)

    # soundData = torch.mean(audio_tensor, dim=0, keepdim=True)
    # tempData = torch.zeros([1, 16000])  # tempData accounts for audio clips that are too short
    # if soundData.numel() < 16000:
    #     tempData[:, :soundData.numel()] = soundData
    # else:
    #     tempData = soundData[:, :16000]

    # soundData = tempData

    # # 计算 MFCC
    # mel_specgram = mel(soundData)
    # mel_specgram = mel_specgram[:,:,:60]
    # mel_specgram = torch.squeeze(mel_specgram)
    # mel_specgram = torch.nan_to_num(mel_specgram, nan=0.0)
    # # 在这里处理或输出 MFCC
    # # print("mfcc: ")
    # # print(mel_specgram.shape)
    # return mel_specgram

def main():
    global recording_audio,recording_video,expressionID,user_name

    # 开始键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    t = threading.Thread(target=record_audio)
    t.daemon = True
    t.start()
    while(1):
        process_audio()
        #continue




if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_normalization",action="store_true",
                        help="Normalize the record data",
                        default=True)

    parser.add_argument("--press_key",action="store_true",
                        help="press 'space' to start the data record",
                        default=True)

    parser.add_argument("--record_data",action="store_true",
                        help="Store landmark data to database",
                        default=False)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    # demo code
    main()