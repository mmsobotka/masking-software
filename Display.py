import numpy as np
from PIL import Image, ImageDraw, ImageColor
import streamlit as st
import cv2
import face_recognition
import mediapipe as mp
import dlib
from FaceDetector import FaceDetector


class Display:
    """
    Display class is responsible for displaying loaded images, video or camera view.
    Displays images after applied modifications.
    """
    @staticmethod
    def load_image_on_sidebar(img):
        image = Image.open(img)
        st.sidebar.image(image)

    @staticmethod
    def load_video(vid):
        video = cv2.VideoCapture(vid.name)
        frames = []

        ret, frame = video.read()
        max_width = 640
        max_height = 480
        height, width, _ = frame.shape

        scale_ratio_1 = height / max_height
        scale_ratio_2 = width / max_width
        scale_ratio = max(scale_ratio_1, scale_ratio_2)
        scale_ratio = 1.0 / scale_ratio

        new_width = int(scale_ratio * width)
        new_height = int(scale_ratio * height)
        dim = (new_width, new_height)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("Can't receive frame - stream end.")
                break
            frame = cv2.resize(frame, dim)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()

        return frames

    @staticmethod
    def view_image(image):
        st.image(image, use_column_width=True)

    @staticmethod
    def get_color(key):
        color = st.sidebar.color_picker("Pick a color", "#FFFFFF", key=key)
        return ImageColor.getcolor(color, "RGB")

    @staticmethod
    def get_slider_size(key, max_value=8.0):
        size = st.sidebar.slider("Select size", 1.0, max_value, 1.0, key=key)
        return size

    @staticmethod
    def draw_box_on_faces(faces, image_to_draw_on, color, use=False):
        if use:
            if len(faces) > 0:
                st.sidebar.success("Face was detected!")
        for face in faces:
            x, y, width, height = face["box"]
            cv2.rectangle(image_to_draw_on, (x, y), (x + width, y + height), color, 2)

    @staticmethod
    def draw_rectangle_under_faces(faces, image_to_draw_on, color):
        for face in faces:
            x, y, width, height = face["box"]
            cv2.rectangle(
                image_to_draw_on,
                (x, y + height),
                (x + width, y + height + 35),
                color,
                cv2.FILLED,
            )

    @staticmethod
    def write_names_under_faces(faces, image_to_draw_on, color, name="UNKNOWN"):
        for face in faces:
            x, y, width, height = face["box"]
            cv2.putText(
                image_to_draw_on,
                name,
                (x + 6, y + height + 28),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                color,
                1,
            )

    @staticmethod
    def draw_points_on_faces(faces, image_to_draw_on, color, size):
        for face in faces:
            for key, value in face["keypoints"].items():
                cv2.circle(image_to_draw_on, value, int(size), color, int(size + 1))

    @staticmethod
    def draw_68_points_on_faces(image_to_draw_on, color, size, landmark_detector):
        faces = FaceDetector.detect_faces_dlib(image_to_draw_on)
        landmark_tuple = []
        for k, d in enumerate(faces):
            landmarks = landmark_detector(image_to_draw_on, d)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmark_tuple.append((x, y))
                cv2.circle(image_to_draw_on, (x, y), int(size), color, -1)
        return image_to_draw_on

    @staticmethod
    def draw_lines_on_faces(image_to_draw_on, color, size):
        face_landmarks_list = face_recognition.face_landmarks(image_to_draw_on)
        pil_image = Image.fromarray(image_to_draw_on)
        size = int(size)
        for face_landmarks in face_landmarks_list:
            d = ImageDraw.Draw(pil_image, "RGBA")
            d.line(face_landmarks["chin"], fill=color, width=size)
            d.line(face_landmarks["left_eyebrow"], fill=color, width=size)
            d.line(face_landmarks["right_eyebrow"], fill=color, width=size)
            d.line(face_landmarks["nose_bridge"], fill=color, width=size)
            d.line(face_landmarks["nose_tip"], fill=color, width=size)
            d.line(face_landmarks["left_eye"], fill=color, width=size)
            d.line(face_landmarks["right_eye"], fill=color, width=size)
            d.line(face_landmarks["top_lip"], fill=color, width=size)
            d.line(face_landmarks["bottom_lip"], fill=color, width=size)
        return np.array(pil_image)

    @staticmethod
    def draw_mesh_on_faces(image_to_draw_on, color, size, mesh_mode):
        mp_draw = mp.solutions.drawing_utils
        draw_spec = mp_draw.DrawingSpec(
            thickness=int(size), circle_radius=int(size), color=color
        )
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)
        results = face_mesh.process(image_to_draw_on)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    image_to_draw_on, faceLms, mesh_mode, draw_spec, draw_spec
                )

    @staticmethod
    def draw_face_features(
            detection_mode,
            image_to_draw_on,
            faces=None,
            detection_mode_colors=None,
            detection_mode_sizes=None,
            detection_mode_mesh=None,
            landmark_detector=None):

        (is_points_selected,
         is_points_dlib_selected,
         is_lines_selected,
         is_mesh_selected,
         ) = detection_mode
        points_color, points_dlib_color, lines_color, mesh_color = detection_mode_colors
        points_size, points_dlib_size, lines_size, mesh_size = detection_mode_sizes

        if is_points_selected:
            Display.draw_points_on_faces(
                faces, image_to_draw_on, points_color, points_size
            )
        if is_points_dlib_selected:
            Display.draw_68_points_on_faces(
                image_to_draw_on, points_dlib_color, points_dlib_size, landmark_detector
            )
        if is_mesh_selected:
            Display.draw_mesh_on_faces(
                image_to_draw_on, mesh_color, mesh_size, detection_mode_mesh
            )
        if is_lines_selected:
            image_to_draw_on = Display.draw_lines_on_faces(
                image_to_draw_on, lines_color, lines_size
            )

        return image_to_draw_on

    @staticmethod
    def get_person_name_label():
        name = st.sidebar.text_input("Person name", " ")
        return name

    @staticmethod
    def write_person_name_on_face(
            recognition_result, faces, image_after_masking, person_name):
        if recognition_result is not None:
            name = "UNKNOWN"
            if recognition_result > 40:
                name = person_name + " " + str(recognition_result) + "%"
            Display.draw_rectangle_under_faces(faces, image_after_masking, (255, 0, 0))
            Display.write_names_under_faces(
                faces, image_after_masking, (255, 255, 255), name
            )
        return image_after_masking
