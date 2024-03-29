from Display import Display
import mediapipe as mp
import streamlit as st

from EnumModeSelector import UploadMode
from EnumModeSelector import FaceDetectionMode
from EnumModeSelector import MaskingOption
from EnumModeSelector import MeshMode


class ModeSelector:
    """
    ModeSelector class is an interface between the user and the system
    that captures the options selected by the user at application runtime.
    """
    sidebar_options = [
        UploadMode.upload_default,
        UploadMode.upload_image,
        UploadMode.upload_video,
        UploadMode.upload_live_camera,
    ]

    @staticmethod
    def load_detection_mode():
        points_color = None
        points_dlib_color = None
        lines_color = None
        mesh_color = None

        points_size = None
        points_dlib_size = None
        lines_size = None
        mesh_size = None

        face_mesh_mode = None

        st.sidebar.write("Set face features as:")
        points = st.sidebar.checkbox("5 points")
        if points:
            points_color = Display.get_color("points")
            points_size = Display.get_slider_size("points")

        points_dlib = st.sidebar.checkbox("68 points")
        if points_dlib:
            points_dlib_color = Display.get_color("points_dlib")
            points_dlib_size = Display.get_slider_size("points_dlib")

        lines = st.sidebar.checkbox("lines")
        if lines:
            lines_color = Display.get_color("lines")
            lines_size = Display.get_slider_size("lines")

        mesh = st.sidebar.checkbox("mesh")
        if mesh:
            face_mesh_mode = ModeSelector.face_mesh_mode()
            mesh_color = Display.get_color("mesh")
            mesh_size = Display.get_slider_size("mesh")

        return (
            (points, points_dlib, lines, mesh),
            (points_color, points_dlib_color, lines_color, mesh_color),
            (points_size, points_dlib_size, lines_size, mesh_size),
            face_mesh_mode,
        )

    @staticmethod
    def load_recognition_mode_check_box():
        face_recognition_mode = st.sidebar.radio(
            "Select mode for face recognition",
            (FaceDetectionMode.cnn, FaceDetectionMode.hog),
        )
        return face_recognition_mode

    @staticmethod
    def load_mask_mode():
        filtration_size = None
        masking_sidebar_options = [
            MaskingOption.default,
            MaskingOption.gaussian_filter,
            MaskingOption.extract_face_features,
            MaskingOption.accurate_extract_face_features,
            MaskingOption.extract_face_features_interpolation,
        ]
        mask_mode = st.sidebar.selectbox(
            "Set masking method as:", masking_sidebar_options
        )
        if mask_mode == MaskingOption.gaussian_filter:
            filtration_size = Display.get_slider_size("Size", max_value=60.0)
        if mask_mode == MaskingOption.extract_face_features_interpolation:
            filtration_size = Display.get_slider_size("Size", max_value=30.0)
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
        mesh_mode = st.sidebar.radio(
            "Set face mesh mode method as",
            (
                MeshMode.mesh_points,
                MeshMode.mesh_contours,
                MeshMode.mesh_triangles,
            ),
        )
        if mesh_mode == MeshMode.mesh_points:
            return mp_face_mesh.FACEMESH_FACE_OVAL
        if mesh_mode == MeshMode.mesh_triangles:
            return mp_face_mesh.FACEMESH_TESSELATION
        if mesh_mode == MeshMode.mesh_contours:
            return mp_face_mesh.FACEMESH_CONTOURS

    @staticmethod
    def load_program_mode():
        program_mode = st.sidebar.selectbox(
            "Choose a target for analysis", ModeSelector.sidebar_options
        )
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
