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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


BATCH_SIZE = 2  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels
EPOCHS = 10
file_name = "Relative_coordinates_data.csv"

df = pd.read_csv('data/{}'.format(file_name),header=None)
X = df.iloc[: , 1:]
y = df.iloc[: , :1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(type(X.values))

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0



print('Train shape:')
print(X_train.shape)
print(X_test.shape)
print('\nTest Shape:')
print(y_train.shape)
print(y_test.shape)


model = tf.keras.models.Sequential([
    
    # tf.keras.layers.Flatten(input_shape=(1434)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')

    # tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(None,1434)),
    # tf.keras.layers.MaxPooling1D(2, 2),

    # tf.keras.layers.Conv1D(64, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2,2),
    
    # tf.keras.layers.Conv1D(128, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2,2),
    
    # tf.keras.layers.Conv1D(128, 3, activation='relu'),
    # tf.keras.layers.MaxPooling1D(2,2),
    
    # tf.keras.layers.GlobalMaxPooling1D(),
    # tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])




model.fit(X_train.values, y_train.values, validation_split=0.1 ,epochs=500)
print("\n")
loss, accuracy = model.evaluate(X_test.values,y_test.values)
print("Accuracy", accuracy)


model.save("Relative_coordinates_model")

# rc = RandomForestClassifier()
# model = rc.fit(X_train, y_train.values.ravel())
# y_pred = model.predict(X_test)

# print(accuracy_score(y_test, y_pred))

