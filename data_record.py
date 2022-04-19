# record the landmark data.
from asyncore import write
import imp
from tkinter import image_names
from traceback import print_tb
import cv2
from matplotlib.pyplot import axis
import mediapipe as mp
import numpy as np
import time
import csv
import keyboard
import os


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


t_end = time.time() + 2

datacheck = 0
stop_time = time.time() + 20
data = []
size_of_landmarks = 478
data_dimentions = 3
np_arr = np.empty((0,size_of_landmarks,data_dimentions))
image_counter = 0
datas = []
file_name_array = []
rows = []
origin_x = 0
origin_y = 0
origin_z = 0

# flags
train = 1
store_method = 1
expression_ID = 1
show_landmarks = 1
relative_coordinates = 1
if relative_coordinates == 1:
    landmark_store_file_name = "Relative_coordinates_data.csv"
else:
    landmark_store_file_name = "simple_data.csv"


#get last image ID
if (train == 1):
# print(os.stat("data/train/expression_{}/file_name_list.csv".format(expression_ID)).st_size)
    if os.stat("data/train/expression_{}/file_name_list.csv".format(expression_ID)).st_size != 0:
        with open("data/train/expression_{}/file_name_list.csv".format(expression_ID), "r") as file:
            print("open file")
            last_line = file.readlines()[-2]
            # print(last_line)
            image_counter = int(list(filter(str.isdigit, last_line))[0]) + 1
            # print(image_counter)
            file.close
else:
    if os.stat("data/test/expression_{}/file_name_list.csv".format(expression_ID)).st_size != 0:
        with open("data/test/expression_{}/file_name_list.csv".format(expression_ID), "r") as file:
            print("open file")
            last_line = file.readlines()[-2]
            # print(last_line)
            image_counter = int(list(filter(str.isdigit, last_line))[0]) + 1
            # print(image_counter)
            file.close

if(train == 1):
    f = open('data/train/expression_{}/landmark_data.csv'.format(expression_ID), 'a')
    file_name_list = open('data/train/expression_{}/file_name_list.csv'.format(expression_ID), 'a')
else:
    f = open('data/test/expression_{}/landmark_data.csv'.format(expression_ID), 'a')
    file_name_list = open('data/test/expression_{}/file_name_list.csv'.format(expression_ID), 'a')

writer = csv.writer(f)
file_name_writer = csv.writer(file_name_list)

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
    results = face_mesh.process(image)
    record_image = image

    if show_landmarks:
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

    # get results.multi_face_landmarks
    if keyboard.is_pressed('space'):

        if store_method == 1:
        # method two(put all data in one csv file)
            face = results.multi_face_landmarks[0].landmark
            if relative_coordinates == 1:
                origin_x = face[0].x
                origin_y = face[0].y
                origin_z = face[0].z
                face_row = list(np.array([[origin_x - landmark.x, origin_y -  landmark.y, origin_z - landmark.z] for landmark in face]).flatten())
            else:
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in face]).flatten())
            row = face_row 
            row.insert(0, expression_ID)
            rows.append(row)


        elif store_method == 0:
            for index in range(len(results.multi_face_landmarks[0].landmark)):
                data.append([results.multi_face_landmarks[0].landmark[index].x,results.multi_face_landmarks[0].landmark[index].y,results.multi_face_landmarks[0].landmark[index].z])
            #writer.writerow(datas)
            if train == 1:
                cv2.imwrite("data/train/expression_{}/images/".format(expression_ID) + image_names,record_image)
            else:
                cv2.imwrite("data/test/expression_{}/images/".format(expression_ID) + image_names,record_image)
            np_arr = np.append(np_arr, np.array([data]),axis = 0)
            datas.append(data)
            data = []
        image_names = "image_{}.png".format(image_counter)
        file_name_array.append([image_names])
        image_counter += 1
        print("store data")
        time.sleep(1)
    
    # save method two
      # k = cv2.waitKey(1)
      # if k%256 == 32:
      #   for index in range(len(results.multi_face_landmarks[0].landmark)):
      #     datas.append([results.multi_face_landmarks[0].landmark[index].x,results.multi_face_landmarks[0].landmark[index].y,results.multi_face_landmarks[0].landmark[index].z])
      #   writer.writerow(datas)
      #   np_arr = np.append(np_arr, np.array([datas]),axis = 0)
      #   datas = []   
      #   print("data save")

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if keyboard.is_pressed('q'):
        break

print("The expression ID is: " + str(expression_ID))
result = input("if you want to store those data:")
if result.find('y') != -1:
    if store_method == 1:
        with open('data/{}'.format(landmark_store_file_name), mode='a', newline='') as f:
            for row in rows:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            for name in file_name_array:
                file_name_writer.writerow(name)
    elif store_method == 0:
        for record_data in datas:
            writer.writerow(record_data)
        for name in file_name_array:
            file_name_writer.writerow(name)
    print("store all data")
else:
    print("delete data")

f.close()
file_name_list.close()
cap.release()

