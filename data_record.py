import os
import cv2
import mediapipe as mp
import time
import keyboard
import numpy as np
import csv
import sounddevice
import sys
import soundfile as sf
import threading
from argparse import ArgumentParser
from core.videosource import WebcamSource
from core.face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


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
frame_step = 28 # vedio record length. default is 28 frame.
floder_name = '' # floder that store user's data
file_3d = ''
file_2d = ''
# set audio record
fs = 16000
seconds = 1


def on_press(key):
    if key == keyboard.Key.space:  # 检测空格键
        # 创建并启动线程，录制音频和视频
        threading.Thread(target=video_record, args=(2,)).start()
        threading.Thread(target=audio_record, args=(2,)).start()
    elif key == keyboard.KeyCode.from_char('q'):  # 检测 'q' 键
        print("Exiting the program")
        cap.release()
        sys.exit()



def main():

  
  # ask user enter info
  user_name = input("please enter your name: ")
  expressionID = input("Please enter the expression ID you want to record: ")
  

  
  # set file name that store 3D and 2D data
  folder_name = user_name
  # create folder if first time record
  if not os.path.exists("data/{}".format(folder_name)):
    os.mkdir("data/{}".format(folder_name))
  

  if args.data_normalization:
    threeDfile = "3D_time_data_{}.csv".format(user_name) 
    twoDfile = "2D_time_data_{}.csv".format(user_name)
  else:
    threeDfile = "3D_time_data_c.csv".format(user_name)
    twoDfile = "2D_time_data_c.csv".format(user_name)
  file_3d = open('data/{}/{}'.format(user_name, threeDfile), mode='a', newline='')
  file_2d = open('data/{}/{}'.format(user_name, twoDfile), mode='a', newline='')


  with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

  
  if cv2.waitKey(1) & 0xFF == 'q': 
    file_3d.close()
    file_2d.close()
    cap.release()


def audio_record():
  print("file name")


def video_record():
  global file_3d, file_2d
   # record start
  flag2= 0
  rows_3d = []
  rows_2d = []
  flag2= 0
  frame_count = 0

  pcf = PCF(
      near=1,
      far=10000,
      frame_height=frame_height,
      frame_width=frame_width,
      fy=camera_matrix[1, 1],
  )

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
      image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_mesh.process(image)

      # Draw the face mesh annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

      # store landmark video
      if args.press_key:
        if keyboard.is_pressed('space'):
          print("start record data")
          start_time = time.time()
          flag2 = 1
          time.sleep(0.1)
      else:
        flag2 = 1

      if flag2 == 1:
        if frame_count < frame_step:
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

          rows_3d.append(face_row_3d)
          rows_2d.append(face_row_2d)
          frame_count += 1   
        else:
          flag2 = 0
          rows_3d.insert(0,expressionID)
          rows_2d.insert(0,expressionID)
          # write 3D data to file
          if args.record_data:
            csv_writer1 =  csv.writer(file_3d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer1.writerow(rows_3d)

            csv_writer2 =  csv.writer(file_2d, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer2.writerow(rows_2d)
            print("record finish, size is: " + str(len(rows_2d)) + " and: " + str(len(rows_2d[1])))

          else:
            print("record finish, size is: " + str(len(rows_2d)) + " and: " + str(len(rows_2d[1])))
            print("record fail")
          rows_3d = []
          rows_2d = []
          frame_count =0
      # Flip the image horizontally for a selfie-view display.
      cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
      cv2.imshow('second window', cv2.flip(image2,1))


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
                        default=True)

    parser.add_argument("--debug", action="store_true",
                        help="showing raw values of detection in the terminal",
                        default=False)

    args = parser.parse_args()

    # demo code
    main()