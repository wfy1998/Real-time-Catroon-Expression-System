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
from collections import deque 
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,random_split
import torchaudio
import torch.optim as optim

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
frame_step = 30 # vedio record length. default is 60 frame.
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

lock = threading.Lock()

def on_press(key):
    global space_press, stop
    if key == keyboard.Key.space:  # 检测空格键
        print("change on press")
        space_press = True
    elif key == keyboard.KeyCode.from_char('q'):
        stop = True
    
def on_release(key):
    global space_press                     
    if key == keyboard.Key.space:
        space_press =False
        print("Space key released!")
    
def audio_record(duration = 0.5):
    global mel_specgram, lock
    # print('Recording audio')
    print("开始录音...")
    sample_rate = 16000
    hop_length = sample_rate // 30
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length-2)
    while(1):
        audio_frames = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        audio_tensor = torch.from_numpy(audio_frames).float().to(torch.float32)
        # print("the audio tensor is: {}".format(audio_tensor.shape))
        soundData = torch.mean(audio_tensor.T, dim=0, keepdim=True)
        # print("the soundData is: {}".format(soundData))
        tempData = torch.zeros([1, 16000])  # tempData accounts for audio clips that are too short
        if soundData.numel() < 16000:
            tempData[:, :soundData.numel()] = soundData
        else:
            tempData = soundData[:, :16000]
        soundData = tempData
        mel_specgram = mel(soundData)
        mel_specgram = mel_specgram[:,:,:30]
        mel_specgram = torch.squeeze(mel_specgram)
        #print("the mel_specgram is: {}".format(mel_specgram.shape))


    
def data_combination(time_data, audio_data):
    numpy_d = np.array(time_data)
    tem = torch.from_numpy(numpy_d)
    # print("tem")
    # print(tem.shape)
    # print(audio_data.shape)
    # print("info")
    # print(tem.shape)
    #print("combination shape: time is {}. audio is {}".format(tem.T.shape, audio_data.shape))
    final_data = torch.cat((tem.T, audio_data),dim=0) # correct 28, x(time, feature)
    # print(final_data.shape)
    final_data = final_data.T
    # print(final_data.shape)
    final_data = final_data.unsqueeze(0)
    # print(final_data.shape)
    return final_data # 1064

def main():
    global recording_audio,recording_video,expressionID,user_name,mel_specgram
    # record vedio
    rows_3d = deque()
    rows_2d = deque()
    frame_count = 0
    record_video = 0
    frame_back_count = 0

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    # ask user enter info
    user_name = input("please enter your name: ")
    expressionID = input("Please enter the expression ID you want to record: ")
    
    # set file name that store 3D and 2D data
    folder_name = user_name
    normal_count = 15
    special_count = 5
    # create folder if first time record
    if not os.path.exists("data/{}".format(folder_name)):
        os.mkdir("data/{}".format(folder_name))
    

    # if args.data_normalization:
    #     threeDfile = "3D_time_data_{}.csv".format(user_name) 
    #     twoDfile = "2D_time_data_{}.csv".format(user_name)
    # else:
    #     threeDfile = "3D_time_data_c.csv".format(user_name)
    #     twoDfile = "2D_time_data_c.csv".format(user_name)
    # file_3d = open('data/{}/{}'.format(user_name, threeDfile), mode='a', newline='')
    # file_2d = open('data/{}/{}'.format(user_name, twoDfile), mode='a', newline='')

    # 开始键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()


    t = threading.Thread(target=audio_record)
    t.daemon = True
    t.start()

    print("open mediapipe windows")
    with mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue 
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))


            if args.data_normalization:
                landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark ])
                landmarks = landmarks.T
                landmarks = landmarks[:, :landmark_number]
                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                        landmarks.copy(), pcf
                    )
                model_points = metric_landmarks[0:3, points_idx].T
            
                origin_x = model_points[0][0]
                origin_y = model_points[0][1]
                origin_z = model_points[0][2]
                face_row_3d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1], origin_z - landmark[2]] for landmark in model_points]).flatten())
                face_row_2d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1]] for landmark in model_points]).flatten())
            else:
                face = results.multi_face_landmarks[0].landmark 
                origin_x = face[0].x
                origin_y = face[0].y
                origin_z = face[0].z
                face_row_3d = list(np.array([[origin_x - landmark.x, origin_y -  landmark.y, origin_z - landmark.z] for landmark in face]).flatten())
                face_row_2d = list(np.array([[origin_x - landmark.x, origin_y -  landmark.y] for landmark in face]).flatten())

            if frame_count < frame_step:
                rows_3d.append(face_row_3d)
                rows_2d.append(face_row_2d)
                frame_count += 1  

            else:
                rows_3d.popleft()
                # time_data.append(relative_face)
                rows_3d.append(face_row_2d)
                
                rows_2d.popleft()
                # time_data.append(relative_face)
                rows_2d.append(face_row_2d)
                # if record the data
                
                # label change to special
                if space_press:
                    label = expressionID
                    frame_back_count = frame_step
                    special_count -= 1
                elif frame_back_count > 0:
                    print("delay")
                    label = expressionID
                    frame_back_count -=1
                    special_count -= 1
                else:
                    label = 0
                    normal_count -= 1
                # keep recording data 
                mel_specgram = torch.squeeze(mel_specgram)
                audio_data = mel_specgram
                video_data = rows_2d
                
                final_data = data_combination(time_data=video_data,audio_data=audio_data)
                #print("final_data shape is {}".format(final_data.shape))
                # 插入label

                # 存进csv file
                if args.record_data:

                    if label == 0 and normal_count <= 0 :
                        print("store 0")
                        normal_count = 15
                        # 将数据和标签转换为列表
                        data_list = final_data.tolist()
                        label_list = [label]
                        # 创建包含数据和标签的 DataFrame
                        df = pd.DataFrame({'label': label_list, 'data': [data_list] })
                        # 将 DataFrame 存储为 CSV 文件
                        csv_filename = 'data/{}/{}_2D_data.csv'.format(user_name,user_name)
                        df.to_csv(csv_filename, mode='a', header=not pd.io.common.file_exists(csv_filename), index=False)

                    elif label == 0:
                        # print("continue no store")
                        pass
                    elif special_count <= 0:
                        print("store special")
                        special_count = 5
                        data_list = final_data.tolist()
                        label_list = [label]
                        # 创建包含数据和标签的 DataFrame
                        df = pd.DataFrame({ 'label': label_list,'data': [data_list]})
                        # 将 DataFrame 存储为 CSV 文件
                        csv_filename = 'data/{}/{}_2D_data.csv'.format(user_name,user_name)
                        df.to_csv(csv_filename, mode='a', header=not pd.io.common.file_exists(csv_filename), index=False)
                    else:
                        pass

                
            # recording data
            # if args.press_key:
            #     if recording_audio:
            #         t = threading.Thread(target=record_audio)
            #         t.daemon = True
            #         t.start()
            #         recording_audio = False
            #     if recording_video:
            #         print("start recording video")
            #         record_video = 1
            #         recording_video = False
            # else: # keep recording data
            #     record_video = 1

            # if record_video == 1: 
            #     if frame_count < frame_step:
            #         if args.data_normalization:
            #             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark ])
            #             landmarks = landmarks.T
            #             landmarks = landmarks[:, :landmark_number]
            #             metric_landmarks, pose_transform_mat = get_metric_landmarks(
            #                     landmarks.copy(), pcf
            #                 )
            #             model_points = metric_landmarks[0:3, points_idx].T
                    
            #             origin_x = model_points[0][0]
            #             origin_y = model_points[0][1]
            #             origin_z = model_points[0][2]
            #             face_row_3d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1], origin_z - landmark[2]] for landmark in model_points]).flatten())
            #             face_row_2d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1]] for landmark in model_points]).flatten())
            #         else:
            #             face = results.multi_face_landmarks[0].landmark 
            #             origin_x = face[0].x
            #             origin_y = face[0].y
            #             origin_z = face[0].z
            #             face_row_3d = list(np.array([[origin_x - landmark.x, origin_y -  landmark.y, origin_z - landmark.z] for landmark in face]).flatten())
            #             face_row_2d = list(np.array([[origin_x - landmark.x, origin_y -  landmark.y] for landmark in face]).flatten())

            #         rows_3d.append(face_row_3d)
            #         rows_2d.append(face_row_2d)
            #         frame_count += 1   
            #     else:
            #         record_video = 0
            #         rows_3d.insert(0,expressionID)
            #         rows_2d.insert(0,expressionID)
            #         # write 3D data to file
            #         if args.record_data:
            #             csv_writer1 =  csv.writer(file_3d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #             csv_writer1.writerow(rows_3d)

            #             csv_writer2 =  csv.writer(file_2d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #             csv_writer2.writerow(rows_2d)
            #             print("record finish, size is: " + str(len(rows_2d)) + " and: " + str(len(rows_2d[1])))

            #         else:
            #             print("record finish, size is: " + str(len(rows_2d)) + " and: " + str(len(rows_2d[1])))
            #             print("record fail")
            #         rows_3d = []
            #         rows_2d = []
            #         frame_count =0
            #         recording_video = False
            if cv2.waitKey(1) & 0xFF == 'q':
                break
        # cv2.imshow('second window', cv2.flip(image2,1))

        # 结束后释放资源
        # file_3d.close()
        # file_2d.close()
        cap.release()



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_normalization",action="store_true",
                        help="Normalize the record data",
                        default=True)

    parser.add_argument("--record_data",action="store_true",
                        help="Store landmark data to database",
                        default=True)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    # demo code
    main()