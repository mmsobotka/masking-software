import streamlit as st

import FaceFilter
from Information import *
from ModeSelector import *
from Display import *
from FaceDetector import *
from FaceFilter import *
import tempfile

#TODO
# when mesh is selected add 2 options - lines, points
# display message - face is detected
# restart button


class Application:
    image_loaded = None
    video_loaded = None
    camera_loaded = None
    image_learn_recognition_loaded = None

    detection_mode = None
    detection_mode_colors = None
    detection_mode_sizes = None
    recognition_mode = None
    masking_mode = None
    masking_size = None
    is_face_detection_enabled = None
    is_face_recognition_enabled = None

    image_after_masking = None
    box_on_faces = None
    faces = None

    def __init__(self):
        Information.print_page_title()

    def load_interface(self):
        program_mode = ModeSelector.load_program_mode()

        if program_mode == ModeSelector.upload_default:
            Information.print_main_page()

        if program_mode == ModeSelector.upload_image:
            self.load_image_mode()
            if self.image_loaded:
                Display.load_image_on_sidebar(self.image_loaded)

        if self.image_loaded or self.video_loaded or self.camera_loaded:
            self.enable_face_detection()
            if self.is_face_detection_enabled:
                self.load_box_on_face_check_box()
                self.load_detection_mode()
                self.load_masking_mode()
                self.run_masking_mode()
                self.detect_faces()
                self.draw_on_image()

            self.enable_face_recognition()
            if self.is_face_recognition_enabled:
                self.load_image_to_learn_recognition_mode()

            Display.view_image(self.image_after_masking)

    def load_image_mode(self):
        image_loaded = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_1")
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_loaded = image
            self.prepare_image_to_draw_on()

    def load_image_to_learn_recognition_mode(self):
        image_loaded = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_2")
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_learn_recognition_loaded = image

    def load_detection_mode(self):
        (self.detection_mode), (self.detection_mode_colors), (
            self.detection_mode_sizes) = ModeSelector.load_detection_mode()

    def load_masking_mode(self):
        mask_mode, size = ModeSelector.load_mask_mode()
        self.masking_mode = mask_mode
        self.masking_size = size

    def enable_face_detection(self):
        is_face_detection_enabled = ModeSelector.load_face_detector_check_box()
        self.is_face_detection_enabled = is_face_detection_enabled

    def enable_face_recognition(self):
        is_face_recognition_enabled = ModeSelector.load_face_recognition_check_box()
        self.is_face_recognition_enabled = is_face_recognition_enabled

    def detect_faces(self):
        self.faces = FaceDetector.detect_faces(self.image_loaded, self.image_after_masking)

    def prepare_image_to_draw_on(self):
        image = cv2.imread(self.image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_after_masking = image_rgb

    def load_box_on_face_check_box(self):
        self.box_on_faces = ModeSelector.load_box_on_face_check_box()

    def load_recognition_mode_check_box(self):
        self.recognition_mode = ModeSelector.load_recognition_mode_check_box()

    def draw_on_image(self):
        if self.box_on_faces:
            Display.draw_box_on_faces(self.faces, self.image_after_masking, (255, 0, 0))

        self.image_after_masking = Display.draw_face_features(self.detection_mode, self.image_after_masking,
                                                              self.faces, self.image_loaded,
                                                              self.detection_mode_colors,
                                                              self.detection_mode_sizes)

    def run_masking_mode(self):
        if self.masking_mode == ModeSelector.default:
            pass
        elif self.masking_mode == ModeSelector.gaussian_filter:
            # FaceFilter.run_face_gausian_filter(self.faces, self.image_after_masking)
            self.image_after_masking = FaceFilter.run_face_gausian_filter(self.image_loaded, self.image_after_masking,
                                                                          self.masking_size,
                                                                          FaceFilter.face_without_forehead_chin_indices)

        elif self.masking_mode == ModeSelector.extract_face_features:
            self.image_after_masking = FaceFilter.run_face_cut_features(self.image_loaded, self.image_after_masking,
                                                                        FaceFilter.left_eye_indices)
            self.image_after_masking = FaceFilter.run_face_cut_features(self.image_loaded, self.image_after_masking,
                                                                        FaceFilter.right_eye_indices)
        elif self.masking_mode == ModeSelector.face_transform:
            pass
        elif self.masking_mode == ModeSelector.extract_face_features_interpolation:
            inerpolation_mode = ModeSelector.load_inerpolation_mode()
            self.run_interpolation_mode(inerpolation_mode)
            # self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
            #   self.image_loaded, self.image_after_masking, )

    def run_interpolation_mode(self, inerpolation_mode):
        right_eye, left_eye, nose, mouth = inerpolation_mode
        if right_eye:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_loaded,
                self.image_after_masking,
                self.masking_size,
                FaceFilter.right_eye_indices)
        if left_eye:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_loaded,
                self.image_after_masking,
                self.masking_size,
                FaceFilter.left_eye_indices)
        if nose:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_loaded,
                self.image_after_masking,
                self.masking_size,
                FaceFilter.nose_indices)
        if mouth:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_loaded,
                self.image_after_masking,
                self.masking_size,
                FaceFilter.mouth_indices)
