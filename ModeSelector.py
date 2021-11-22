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
    extract_face_features = "Cut face features"
    face_transform = "Face transform"
    extract_face_features_interpolation = "Extract face features interpolation"

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
        filtration_size = None
        masking_sidebar_options = [ModeSelector.default, ModeSelector.gaussian_filter,
                                   ModeSelector.extract_face_features, ModeSelector.face_transform,
                                   ModeSelector.extract_face_features_interpolation]
        mask_mode = st.sidebar.selectbox("Set masking method as:", masking_sidebar_options)
        if mask_mode == ModeSelector.gaussian_filter:
            filtration_size = Display.get_slider_size("size", max=60.0)
        if mask_mode == ModeSelector.extract_face_features_interpolation:
            filtration_size = Display.get_slider_size("size", max=30.0)
        return mask_mode, filtration_size

    @staticmethod
    def load_inerpolation_mode():
        right_eye = st.sidebar.checkbox("right_eye")
        left_eye = st.sidebar.checkbox("left eye")
        nose = st.sidebar.checkbox("nose")
        mouth = st.sidebar.checkbox("mouth")

        return right_eye, left_eye, nose, mouth

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
