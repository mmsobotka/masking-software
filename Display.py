import numpy as np
from PIL import Image, ImageDraw, ImageColor
import streamlit as st
import cv2
import face_recognition
import mediapipe as mp

class Display:
    @staticmethod
    def load_image_on_sidebar(img):
        image = Image.open(img)
        st.sidebar.image(image)

    @staticmethod
    def view_image(image):
        st.image(image, use_column_width=True)

    @staticmethod
    def get_color(key):
        """TODO pass key to info"""
        color = st.sidebar.color_picker("Pick a color", "#FFFFFF", key=key)
        return ImageColor.getcolor(color, "RGB")

    @staticmethod
    def get_slider_size(key):
        size = st.sidebar.slider("Select size", 0.1, 10.0, 2.0, key=key)
        return size

    @staticmethod
    def draw_box_on_faces(faces, image_to_draw_on, color):
        for face in faces:
            x, y, width, height = face['box']
            cv2.rectangle(image_to_draw_on, (x, y), (x + width, y + height), color, 2)

    @staticmethod
    def draw_points_on_faces(faces, image_to_draw_on, color, size):
        for face in faces:
            for key, value in face['keypoints'].items():
                cv2.circle(image_to_draw_on, value, int(size), color, int(size+1))

    @staticmethod
    def draw_lines_on_faces(image_loaded, image_to_draw_on, color, size):
        image = cv2.imread(image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(image_rgb)
        pil_image = Image.fromarray(image_to_draw_on)
        size = int(size)
        for face_landmarks in face_landmarks_list:
            d = ImageDraw.Draw(pil_image, 'RGBA')
            d.line(face_landmarks['chin'], fill=color, width=size)
            d.line(face_landmarks['left_eyebrow'], fill=color, width=size)
            d.line(face_landmarks['right_eyebrow'], fill=color, width=size)
            d.line(face_landmarks['nose_bridge'], fill=color, width=size)
            d.line(face_landmarks['nose_tip'], fill=color, width=size)
            d.line(face_landmarks['left_eye'], fill=color, width=size)
            d.line(face_landmarks['right_eye'], fill=color, width=size)
            d.line(face_landmarks['top_lip'], fill=color, width=size)
            d.line(face_landmarks['bottom_lip'], fill=color, width=size)

        return np.array(pil_image)

    @staticmethod
    def draw_mesh_on_faces(image_loaded, image_to_draw_on, color, size):
        image = cv2.imread(image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_draw = mp.solutions.drawing_utils
        draw_spec = mp_draw.DrawingSpec(thickness=int(size), circle_radius=int(size), color=color)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)

        # imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mp_draw.draw_landmarks(image_to_draw_on, faceLms, mp_face_mesh.FACEMESH_TESSELATION, draw_spec,
                                       draw_spec)

    @staticmethod
    def draw_face_features(detection_mode, image_to_draw_on, faces=None, image_loaded=None, detection_mode_colors=None,
                           detection_mode_sizes=None):

        is_points_selected, is_lines_selected, is_mesh_selected = detection_mode
        points_color, lines_color, mesh_color = detection_mode_colors
        points_size, lines_size, mesh_size = detection_mode_sizes

        if is_points_selected:
            Display.draw_points_on_faces(faces, image_to_draw_on, points_color, points_size)
        if is_mesh_selected:
            Display.draw_mesh_on_faces(image_loaded, image_to_draw_on, mesh_color, mesh_size)
        if is_lines_selected:
            image_to_draw_on = Display.draw_lines_on_faces(image_loaded, image_to_draw_on, lines_color, lines_size)

        return image_to_draw_on
