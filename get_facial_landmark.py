"""
For finding the face and face landmarks for further manipulication
"""

import cv2
import mediapipe as mp
import numpy as np
from core.videosource import WebcamSource

from core.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)


# normalization
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# uncomment next line to use all points for PnP algorithm
points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1]
points_idx = list(range(0,468))
frame_height, frame_width, channels = (720, 1280, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

dist_coeff = np.zeros((4, 1))




class FaceMeshDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_faces=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.origin_x = 0
        self.origin_y = 0
        self.origin_z = 0

        # Facemesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # The object to do the stuffs
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_image_mode,
            self.max_num_faces,
            True,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):

        pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],)

        # convert the img from BRG to RGB
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        img.flags.writeable = False
        self.results = self.face_mesh.process(img)

        # Draw the face mesh annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = img.shape

        self.faces = []
        self.relative_face = []
        self.relative_face_2D = []
        self.threeD_face = []
        self.face_row_3d = []
        self.face_row_2d = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:

                if draw:
                    self.mp_drawing.draw_landmarks(
                        image = img,
                        landmark_list = face_landmarks,
                        connections = self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec = self.drawing_spec,
                        connection_drawing_spec = self.drawing_spec)
                # generate relative face data
                face_landmarks = self.results.multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                # print(landmarks.shape)
                landmarks = landmarks.T
                landmarks = landmarks[:, :468]

                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )

                image_points = (
                    landmarks[0:2, points_idx].T
                    * np.array([frame_width, frame_height])[None, :]
                )
                model_points = metric_landmarks[0:3, points_idx].T
                origin_x = model_points[0][0]
                origin_y = model_points[0][1]
                origin_z = model_points[0][2]
                self.face_row_3d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1], origin_z - landmark[2]] for landmark in model_points]).flatten())
                self.face_row_2d = list(np.array([[origin_x - landmark[0], origin_y -  landmark[1]] for landmark in model_points]).flatten())

                # xyz_face = self.results.multi_face_landmarks[0].landmark
                # self.origin_x = xyz_face[0].x
                # self.origin_y = xyz_face[0].y
                # self.origin_z = xyz_face[0].z
                # self.relative_face = np.array([[self.origin_x - landmark.x,self.origin_y - landmark.y,self.origin_z - landmark.z] for landmark in xyz_face])
                # self.relative_face_2D = np.array([[self.origin_x - landmark.x,self.origin_y - landmark.y] for landmark in xyz_face])
                # self.threeD_face = np.array([[landmark.x,landmark.y,landmark.z] for landmark in xyz_face])
                # self.relative_face = self.relative_face.flatten()
                # self.relative_face_2D = self.relative_face_2D.flatten()

                face = []
                for id, lmk in enumerate(face_landmarks.landmark):
                    x, y = int(lmk.x * self.imgW), int(lmk.y * self.imgH)
                    face.append([x, y])

                    # show the id of each point on the image
                    # cv2.putText(img, str(id), (x-4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                self.faces.append(face)
        return img, self.faces,self.face_row_3d,self.face_row_2d


# sample run of the module
def main():

    detector = FaceMeshDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        img, faces = detector.findFaceMesh(img)

        # if faces:
        #     print(faces[0])

        cv2.imshow('MediaPipe FaceMesh', img)

        # press "q" to leave
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    # demo code
    main()
