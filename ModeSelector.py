import streamlit as st
from Display import *
import mediapipe as mp


class ModeSelector:
    upload_default = "Mode Selection"
    upload_image = "Load Image"
    upload_video = "Load Video"
    upload_live_camera = "Live Camera"
    cnn_mode = "CNN"
    hog_mode = "HOG"
    # lbph_mode = "LBPH"
    mesh_points_mode = "mesh with points"
    mesh_contours_mode = "mesh with contours"
    mesh_triangles_mode = "mesh with triangles"

    sidebar_options = [upload_default, upload_image, upload_video, upload_live_camera]
    default = "No mask"
    gaussian_filter = "Gaussian filter"
    extract_face_features = "Cut features"
    accurate_extract_face_features = "Accurate cut features"
    # face_transform = "Face transform"
    extract_face_features_interpolation = "Interpolation features"

    @staticmethod
    def load_detection_mode():
        points_color = None
        lines_color = None
        mesh_color = None

        points_size = None
        lines_size = None
        mesh_size = None

        face_mesh_mode = None

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
            face_mesh_mode = ModeSelector.face_mesh_mode()
            mesh_color = Display.get_color("mesh")
            mesh_size = Display.get_slider_size("mesh")

        return (points, lines, mesh), (points_color, lines_color, mesh_color), (
            points_size, lines_size, mesh_size), face_mesh_mode

    @staticmethod
    def load_recognition_mode_check_box():
        face_recognition_mode = st.sidebar.radio("Select mode for face recognition", (
            ModeSelector.cnn_mode, ModeSelector.hog_mode))  # ModeSelector.lbph_mode))
        return face_recognition_mode

    @staticmethod
    def load_mask_mode():
        filtration_size = None
        masking_sidebar_options = [ModeSelector.default, ModeSelector.gaussian_filter,
                                   ModeSelector.extract_face_features, ModeSelector.accurate_extract_face_features,
                                   # ModeSelector.face_transform,
                                   ModeSelector.extract_face_features_interpolation]
        mask_mode = st.sidebar.selectbox("Set masking method as:", masking_sidebar_options)
        if mask_mode == ModeSelector.gaussian_filter:
            filtration_size = Display.get_slider_size("Size", max=60.0)
        if mask_mode == ModeSelector.extract_face_features_interpolation:
            filtration_size = Display.get_slider_size("Size", max=30.0)
        return mask_mode, filtration_size

    @staticmethod
    def load_interpolation_mode():
        right_eye = st.sidebar.checkbox("Right eye")
        left_eye = st.sidebar.checkbox("Left eye")
        nose = st.sidebar.checkbox("Nose")
        mouth = st.sidebar.checkbox("Mouth")

        return right_eye, left_eye, nose, mouth

    @staticmethod
    def face_mesh_mode():
        mp_face_mesh = mp.solutions.face_mesh
        mesh_mode = st.sidebar.radio("Set face mesh mode method as", (
            ModeSelector.mesh_points_mode, ModeSelector.mesh_contours_mode, ModeSelector.mesh_triangles_mode))
        if mesh_mode == ModeSelector.mesh_points_mode:
            return mp_face_mesh.FACEMESH_FACE_OVAL
        if mesh_mode == ModeSelector.mesh_triangles_mode:
            return mp_face_mesh.FACEMESH_TESSELATION
        if mesh_mode == ModeSelector.mesh_contours_mode:
            return mp_face_mesh.FACEMESH_CONTOURS

    @staticmethod
    def load_program_mode():
        program_mode = st.sidebar.selectbox("Choose a target for analysis", ModeSelector.sidebar_options)
        return program_mode

    @staticmethod
    def load_face_detector_check_box():
        return st.sidebar.checkbox("Face detection")

    @staticmethod
    def load_face_recognition_check_box():
        return st.sidebar.checkbox("Face recognition")

    @staticmethod
    def load_box_on_face_check_box():
        return st.sidebar.checkbox("Detect face")
