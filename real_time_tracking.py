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
from pynput import keyboard
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


import numpy as np
import sounddevice as sd
import torch
import torchaudio
import queue
import threading
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,random_split
import torchaudio
import torch.optim as optim



# model
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

# global variable
port = 5066         # have to be same as unity
EMOTIONS = ["natural","expression_1","expression2","expression3","expression4","expression5","expression6"]

user_name = input("please enter your name:")
stop = False

# audio setting
# 设置音频采样率和每个数据块的持续时间
sample_rate = 16000  # 采样率
chunk_duration = 1 # 持续时间（以秒为单位）
chunk_samples = int(sample_rate * chunk_duration)  # 每个数据块的样本数
hop_length = sample_rate //60
duration = 1 # audio record time second
# 初始化 MFCC 转换器
mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=12)

# 创建一个队列来存储音频数据
audio_queue = queue.Queue()

# model para
device = torch.device("cpu")
model = torch.load('models/{}'.format(user_name))
model.to(device)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping
hyperparameters = {"lr": 0.005, "weight_decay": 0.0001, "batch_size": 1, "in_feature": 1064, "out_feature": 7}
optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
log_interval = 10
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}  # needed for using datasets on gpu
# record para
error_key = False
train_data_set = []

# def audio_callback(indata, frames, time, status):
#     # 将音频数据放入队列
#     audio_queue.put(indata.copy())

# def stream_thread_function(audio_callback, sample_rate, chunk_samples):
#     with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=chunk_samples):
#         while True:
#             pass  # Or do some processing related to the stream, if required



def audio_record():
    global audio_frames, mel_specgram
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, hop_length=hop_length-2)
    while(1):
        audio_frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
        sd.wait()
        #print("audio_frames shape is: {} and type is {}".format(audio_frames.shape, type(audio_frames))) #记录会出现nan值
        # has_nan = np.isnan(audio_frames)
        # if np.any(has_nan):
        #     print("audio_record has nan")
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
        # print("the mel_specgram is: {}".format(mel_specgram))
        mel_specgram = mel_specgram[:,:,:60]
        mel_specgram = torch.squeeze(mel_specgram)



# def process_audio():
#     global mel_specgram
#     while(1):
#         audio_frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2)
#         sd.wait()
#         #print("audio_frames shape is: {} and type is {}".format(audio_frames.shape, type(audio_frames))) #记录会出现nan值
#         # has_nan = np.isnan(audio_frames)
#         # if np.any(has_nan):
#         #     print("audio_record has nan")
#         audio_tensor = torch.from_numpy(audio_frames).float().to(torch.float32)
#         # print("the audio tensor is: {}".format(audio_tensor.shape))
#         soundData = torch.mean(audio_tensor.T, dim=0, keepdim=True)
#         # print("the soundData is: {}".format(soundData))
#         tempData = torch.zeros([1, 16000])  # tempData accounts for audio clips that are too short
#         if soundData.numel() < 16000:
#             tempData[:, :soundData.numel()] = soundData
#         else:
#             tempData = soundData[:, :16000]

#         soundData = tempData

#         # write('data/{}/{}_{}.wav'.format(user_name,str(4), str(audio_ID)),  16000, audio_frames)  # 保存为WAV文件
#         # audio_ID +=1
#         mel_specgram = mel(soundData)
#         # print("the mel_specgram is: {}".format(mel_specgram))
#         mel_specgram = mel_specgram[:,:,:60]
#         mel_specgram = torch.squeeze(mel_specgram)
    

# init TCP connection with unity
# return the socket connected
def init_TCP():
    port = args.port

    # '127.0.0.1' = 'localhost' = your computer internal data transmission IP
    address = ('127.0.0.1', port)
    # address = ('192.168.0.107', port)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(address)
        # print(socket.gethostbyname(socket.gethostname()) + "::" + str(port))
        print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(port))
        return s
    except OSError as e:
        print("Error while connecting :: %s" % e)

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # print(socket.gethostbyname(socket.gethostname()))
    # s.connect(address)
    # return s

def send_info_to_unity(s, args):
    msg = '%.4f ' * len(args) % args

    try:
        s.send(bytes(msg, "utf-8"))
    except socket.error as e:
        print("error while sending :: " + str(e))

        # quit the script if connection fails (e.g. Unity server side quits suddenly)
        sys.exit()

def print_debug_msg(args):
    msg = '%.4f ' * len(args) % args
    print(msg)


def on_press(key):
    global error_key, stop
    if key == keyboard.Key.space:  # 检测空格键
        print("change on press")
        error_key = True
    elif key == keyboard.KeyCode.from_char('q'):
        # 当按下 'q' 键，停止监听器
        if args.retrain:
            print("stop--------------")
        # sys.exit()
        stop = True


def store_data(audio_data, video_data ,output):
    global user_name, error_key, train_data_set
    # video data only----------------
    # if len(data) != 60:
    #     print("store data fail since the data length is less than 28 frame.")
    #     return
    # threeDfile = "3D_time_data_feedback.csv"  
    # twoDfile = "2D_time_data_feedback.csv" 
    # file_3d = open('data/{}'.format(threeDfile), mode='a', newline='')
    # file_2d = open('data/{}'.format(twoDfile), mode='a', newline='')

    # data.insert(0,label)
    
    # csv_writer1 =  csv.writer(file_3d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # csv_writer1.writerow(data)

    # csv_writer2 =  csv.writer(file_2d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # csv_writer2.writerow(data)

    # file_3d.close()
    # file_2d.close()

    # new store logic-------------------------------
    # if len(data) != 60:
    #     print("store data fail since the data length is less than 60 frame. the size is {} ".format(len(data)))
    #     return
    threeDfile = "3D_time_data_{}_feedback.csv".format(user_name)  
    twoDfile = "2D_time_data_{}_feedback.csv".format(user_name) 
    file_3d = open('data/{}/{}'.format(user_name,threeDfile), mode='a', newline='')
    file_2d = open('data/{}/{}'.format(user_name,twoDfile), mode='a', newline='')

    if error_key:
        # print("record error data")
        label = torch.max(output, dim=1).indices
        train_data = data_combination(video_data, audio_data)
        train_data = torch.squeeze(train_data)
        label = torch.squeeze(label)
        train_data_set.append((train_data,label))
        error_key = False
    else:
        # print("record correct data")
        label = torch.max(output, dim=1).indices
        # np_audio = audio_data.numpy().T
        # np_video = np.array(video_data)
        # print(np_audio.shape)
        # print(np_video.shape)
        # record_data = np.concatenate((np_audio,np_video), axis = 1)
        # # record_data = np.insert(record_data,0,label)
        # print(record_data.shape)
        # csv_writer1 =  csv.writer(file_3d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer1.writerow(data)

        # csv_writer2 =  csv.writer(file_2d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # csv_writer2.writerow(record_data)

        label = torch.max(output, dim=1).indices
        train_data = data_combination(video_data, audio_data)
        train_data = torch.squeeze(train_data)
        # print("info2")
        # print(train_data.shape)
        numpy_int64 = np.int64(label.item())
        train_data_set.append((train_data,numpy_int64))

    file_3d.close()
    file_2d.close()
    return

def retrain():
    global train_data_set,model
    print("retrain start----")
    print(len(train_data_set))
    train_set = Dataload(user_name=user_name)
    # print("train set size: " + str(len(train_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
    for epoch in range(1, 10):
        train(model=model, epoch=epoch, train_loader=train_loader)
        test(model=model, epoch=epoch, test_loader=train_loader)
    return



def train(model, epoch, train_loader):
    print("training---------------")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print("data shape is {}".format(data.shape))
        data = data.to(device)
        target = target.to(device)
        # print("data size: " + str(data.size()))
        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        # print("the out put is {}".format(output))
        # print("the target is {}".format(target))
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if batch_idx % log_interval == 0: #print training stats
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss))

def test(model, epoch, test_loader):
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

class Dataload(Dataset):
    def __init__(self, user_name):
        global train_data_set
        # load landmark data.
        self.data = []
        self.data = train_data_set
        
        # print(type(transposed_data[0][0]))
        # print(type(int(label)))

        # load audio data
        
 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def classification(audio_data, video_data ,model):
    global device
    label = 0
    # combine audio data and video data
    final_data = data_combination(video_data,audio_data)
    final_data = final_data.to(device)

    #---------------Pytorch LSTM--------------------
    output, hidden_state = model(final_data, model.init_hidden(hyperparameters["batch_size"]))
    pred = torch.max(output, dim=1).indices
    label = pred
    #print(label)
    print(output[0][0:3])
    # --------------tensorflow alexnet-------------------
    # if not data:
    #     return 0
    # arr = np.array(data)
    # if arr.shape[0] != 28:
    #     return 0
    # # print(arr.shape)
    # row = arr.reshape(1,28,936,1)
    # # print(row.shape)
    # result = model.predict(row)
    # emotion_probability = np.max(result)
    # label = EMOTIONS[result.argmax()]
    # # print(result)s
    # if(result[0][1] > 0.50):
    #     label = 1
    #     # print("expression_1")
    # elif(result[0][2] > 0.90):
    #     label = 2
    #     # print("expression_2")
    # elif(result[0][3] > 0.90):
    #     label = 3
    # elif(result[0][4] > 0.90):
    #     label = 4
    # elif(result[0][5] > 0.90):
    #     label = 5
    # elif(result[0][6] > 0.90):
    #     label = 6    
    # else:
    #     label = 0
        # print("expression_2")                     
        # print("expression_0")
    # if keyboard.is_pressed('space'):
    #     # cannot detect the right expression 
    #     print(result[0][1:].argmax() + 1)
    #     label = result[0][1:].argmax() + 1
    #     if args.feedback:
    #         store_data(data, label)
    #     # label = EMOTIONS[result[1:].argmax()]
    # print(label)
    if args.feedback:
        store_data(audio_data, video_data, output)

    return label, output

def data_combination(time_data, audio_data):
    numpy_d = np.array(time_data)
    tem = torch.from_numpy(numpy_d)
    # print("tem")
    # print(tem.shape)
    # print(audio_data.shape)
    # print("info")
    # print(tem.shape)
    final_data = torch.cat((tem.T, audio_data),dim=0) # correct 28, x(time, feature)
    # print(final_data.shape)
    final_data = final_data.T
    # print(final_data.shape)
    final_data = final_data.unsqueeze(0)
    # print(final_data.shape)
    return final_data # 1064

def draw_result(labels):
    print("drawing.....")
    plt.plot(labels)
    plt.show()

def main():
    global user_name, model, stop, mel_specgram
    # Classification data
    classification_result = 0
    time_data = deque()
    frame = 60
    frame_count = 0

    # 开始键盘监听
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # stream_thread = threading.Thread(target=stream_thread_function, args=(audio_callback, sample_rate, chunk_samples))
    # stream_thread.start()
    t = threading.Thread(target=audio_record)
    t.daemon = True
    t.start()
    # use internal webcam/ USB camera
    cap = cv2.VideoCapture(args.cam)
    
    label_one = []
    # IP cam (android only), with the app "IP Webcam"
    # url = 'http://192.168.0.102:4747/video'
    # url = 'https://192.168.0.102:8080/video'
    # cap = cv2.VideoCapture(url)

    # model = torch.load('models/{}'.format(user_name))

    # Facemesh
    detector = FaceMeshDetector()

    # get a sample frame for pose estimation img
    success, img = cap.read()

    # Pose estimation related
    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))
    image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))

    # extra 10 points due to new attention model (in iris detection)
    iris_image_points = np.zeros((10, 2))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for eyes
    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for mouth_dist
    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )


    # Initialize TCP connection
    if args.connect:
        socket = init_TCP()

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose

        # first two steps
        img_facemesh, faces, face_row_3d,face_row_2d = detector.findFaceMesh(img)

        # data normalization
        if len(face_row_2d) == 0:
            continue

        # generate relative time serier data
        if frame_count < frame:
            # time_data.append(relative_face)
            time_data.append(face_row_2d)
            frame_count += 1
        else:
            time_data.popleft()
            # time_data.append(relative_face)
            time_data.append(face_row_2d)

            # get audio data
            audio_data = mel_specgram
            


            # get classification result
            classification_result, confidence_scores = classification(audio_data= audio_data, video_data= time_data, model=model)
            label_one.append(int(confidence_scores[0][1]))

        # flip the input image so that it matches the facemesh stuff
        img = cv2.flip(img, 1)

        # if there is any face detected
        if faces:
            # only get the first face
            for i in range(len(image_points)):
                image_points[i, 0] = faces[0][i][0]
                image_points[i, 1] = faces[0][i][1]
                
            # for refined landmarks around iris
            for j in range(len(iris_image_points)):
                iris_image_points[j, 0] = faces[0][j + 468][0]
                iris_image_points[j, 1] = faces[0][j + 468][1]

            # The third step: pose estimation
            # pose: [[rvec], [tvec]]
            pose = pose_estimator.solve_pose_by_all_points(image_points)

            x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.LEFT)
            x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(image_points, iris_image_points, Eyes.RIGHT)


            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            mouth_distance = FacialFeatures.mouth_distance(image_points)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()

            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            steady_pose = np.reshape(steady_pose, (-1, 3))

            # stabilize the eyes value
            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0])

            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0]

            # uncomment the rvec line to check the raw values
            # print("rvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[0][0], steady_pose[0][1], steady_pose[0][2]))
            # print("tvec steady (x, y, z) = (%f, %f, %f): " % (steady_pose[1][0], steady_pose[1][1], steady_pose[1][2]))

            # calculate the roll/ pitch/ yaw
            # roll: +ve when the axis pointing upward
            # pitch: +ve when we look upward
            # yaw: +ve when we look left
            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
            yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)


            # send info to unity
            if args.connect:

                # for sending to live2d model (Hiyori)
                send_info_to_unity(socket,
                    (roll, pitch, yaw,
                    ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                    mar, mouth_distance,classification_result)
                )

            # print the sent values in the terminal
            if args.debug:
                print_debug_msg((roll, pitch, yaw,
                        ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right,
                        mar, mouth_distance,classification_result))


            # pose_estimator.draw_annotation_box(img, pose[0], pose[1], color=(255, 128, 128))

            # pose_estimator.draw_axis(img, pose[0], pose[1])

            pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        else:
            # reset our pose estimator
            pose_estimator = PoseEstimator((img_facemesh.shape[0], img_facemesh.shape[1]))

        cv2.imshow('Facial landmark', img_facemesh)
        if stop:
            print("tracking sys stop...")
            break
        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if args.retrain:
        retrain()
    if args.draw_result:
        draw_result(label_one)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--data_normalization", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    parser.add_argument("--connect", action="store_true",
                        help="connect to unity character",
                        default=False)

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
                        default=True)
    parser.add_argument("--retrain", action="store_true",
                        help="collect wrong result and store in dataset",
                        default=False)
    parser.add_argument("--draw_result", action="store_true",
                        help="collect wrong result and store in dataset",
                        default=True)
    args = parser.parse_args()

    # demo code
    main()
