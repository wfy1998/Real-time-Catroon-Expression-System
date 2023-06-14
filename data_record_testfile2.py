import cv2
import sounddevice as sd
import numpy as np
from pynput import keyboard
import threading
from scipy.io.wavfile import write

# 创建 VideoWriter 对象，这将用于将来写入视频文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

# 定义全局变量以存储视频和音频
video_frames = []
audio_frames = []
audio_ID = 0

def record_video(duration=2):
    global video_frames
    for i in range(100):
        print("recording video")

def record_audio(duration=2, fs=44100):
    global audio_frames,audio_ID
    audio_frames = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    write('data/{}_{}.wav'.format(str(0), str(audio_ID)),  fs, audio_frames)  # 保存为WAV文件
    audio_ID +=1

def on_press(key):
    if key == keyboard.Key.space:
        threading.Thread(target=record_video).start()
        threading.Thread(target=record_audio).start()
    elif key == keyboard.KeyCode.from_char('q'):
        # 当按下 'q' 键，停止监听器
        return False

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
