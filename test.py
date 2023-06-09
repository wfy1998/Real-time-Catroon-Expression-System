from pynput import keyboard
import cv2
import threading
import wave
import numpy as np
import sys

# 设置音频参数
chunk = 1024  # 记录的块大小
channels = 2  # 双声道
fs = 44100  # 记录频率

frames = []  # 用于储存音频帧的列表
 
# 初始化opencv
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # 输出文件名，编码，帧率，大小

def on_press(key):
    if key == keyboard.Key.space:  # 检测空格键
        # 创建并启动线程，录制音频和视频
        threading.Thread(target=record_audio, args=(2,)).start()
        threading.Thread(target=record_video, args=(2,)).start()
    elif key == keyboard.KeyCode.from_char('q'):  # 检测 'q' 键
        print("Exiting the program")
        cap.release()
        out.release()
        sys.exit()


def record_audio(duration):
    count = 1
    while(1):
        print('Recording audio')
        count += 1
        if count == 10:
            break




def record_video(duration):
    count = 1
    while(1):
        print('Recording video')
        count += 1
        if count == 10:
            break


with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
