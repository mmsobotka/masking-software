import streamlit as st

from Information import *
from ModeSelector import *
from Display import *
from FaceDetector import *
from FaceFilter import *
import tempfile


class Application:
    image_loaded = None
    video_loaded = None
    camera_loaded = None
    detection_mode = None
    detection_mode_colors = None
    detection_mode_sizes = None
    masking_mode = None
    is_face_detection_enabled = None

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
                self.detect_faces()
                self.run_masking_mode()
                self.image_after_masking = FaceFilter.cp(self.image_loaded, self.image_after_masking)
                self.draw_on_image()

            Display.view_image(self.image_after_masking)

    def load_image_mode(self):
        image_loaded = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_1")
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_loaded = image
            self.prepare_image_to_draw_on()

    def load_detection_mode(self):
        (self.detection_mode), (self.detection_mode_colors), (
            self.detection_mode_sizes) = ModeSelector.load_detection_mode()

    def load_masking_mode(self):
        mask_mode = ModeSelector.load_mask_mode()
        self.masking_mode = mask_mode

    def enable_face_detection(self):
        is_face_detection_enabled = ModeSelector.load_face_detector_check_box()
        self.is_face_detection_enabled = is_face_detection_enabled

    def detect_faces(self):
        self.faces = FaceDetector.detect_faces(self.image_loaded)

    def prepare_image_to_draw_on(self):
        image = cv2.imread(self.image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_after_masking = image_rgb

    def load_box_on_face_check_box(self):
        self.box_on_faces = ModeSelector.load_box_on_face_check_box()

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
            FaceFilter.run_face_gausian_filter(self.faces, self.image_after_masking)
        elif self.masking_mode == ModeSelector.extract_face_features:
            pass
        elif self.masking_mode == ModeSelector.face_transform:
            pass



