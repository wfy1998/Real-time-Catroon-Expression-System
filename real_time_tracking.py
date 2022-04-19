import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tkinter import image_names
from traceback import print_tb
import cv2
from matplotlib.pyplot import axis
import mediapipe as mp
import keyboard
import socket

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
EMOTIONS = ["natural" ,"anger"]
show_landmarks = 1

origin_x = 0
origin_y = 0
origin_z = 0
relative_coordinates = 1

# Establish a TCP connection to unity.
address = ('127.0.0.1', 5066)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(address)

reconstructed_model = tf.keras.models.load_model("Relative_coordinates_model")

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
    face = results.multi_face_landmarks[0].landmark
    # print("x : " + str(face[0].x) + "Y : " + str(face[0].y) + "Z : " + str(face[0].z) )
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

    if results.multi_face_landmarks:
      face = results.multi_face_landmarks[0].landmark
      if relative_coordinates == 1:
        origin_x = face[0].x
        origin_y = face[0].y
        origin_z = face[0].z
        face_row = np.array([[origin_x - landmark.x,origin_y - landmark.y,origin_z - landmark.z] for landmark in face])
      else:  
        face_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face])
      row = face_row.reshape(1,1434)
      result = reconstructed_model.predict(row)
      emotion_probability = np.max(result)
      label = EMOTIONS[result.argmax()]
      # print(result)
      # print(emotion_probability)

      # send data to Unity
      msg = label
      s.send(bytes(msg,"utf-8"))
      print(label)
        

    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    if keyboard.is_pressed('q'):
        break

