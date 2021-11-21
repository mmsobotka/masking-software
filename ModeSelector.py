import streamlit as st
from Display import *


class ModeSelector:
    upload_default = "Mode Selection"
    upload_image = "Load Image"
    upload_video = "Load Video"
    upload_live_camera = "Live Camera"

    sidebar_options = [upload_default, upload_image, upload_video, upload_live_camera]

    default = "No mask"
    gaussian_filter = "Gaussian filter"
    extract_face_features = "Extract face features"
    face_transform = "Face transform"

    @staticmethod
    def load_detection_mode():
        points_color = None
        lines_color = None
        mesh_color = None

        points_size = None
        lines_size = None
        mesh_size = None

        st.sidebar.write("Set face features as:")
        points = st.sidebar.checkbox("points")
        if points:
            points_color = Display.get_color("points")
            points_size = Display.get_slider_size("points")

        lines = st.sidebar.checkbox("lines")
        if lines:
            lines_color = Display.get_color("lines")
            lines_size = Display.get_slider_size("lines")

        mesh = st.sidebar.checkbox("mesh")
        if mesh:
            mesh_color = Display.get_color("mesh")
            mesh_size = Display.get_slider_size("mesh")

        return (points, lines, mesh), (points_color, lines_color, mesh_color), (points_size, lines_size, mesh_size)

    @staticmethod
    def load_mask_mode():
        masking_sidebar_options = [ModeSelector.default, ModeSelector.gaussian_filter, ModeSelector.extract_face_features, ModeSelector.face_transform]
        mask_mode = st.sidebar.selectbox("Set masking method as:", masking_sidebar_options)
        return mask_mode

    @staticmethod
    def load_program_mode():
        program_mode = st.sidebar.selectbox("Choose a target for analysis", ModeSelector.sidebar_options)
        return program_mode

    @staticmethod
    def load_face_detector_check_box():
        return st.sidebar.checkbox("Face detection")

    @staticmethod
    def load_box_on_face_check_box():
        return st.sidebar.checkbox("Draw box")
