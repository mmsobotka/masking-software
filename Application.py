import cv2.cv2
import gc

import streamlit as st
from Information import Information
from FaceDetector import FaceDetector
from FaceRecognizer import FaceRecognizer
from ModeSelector import ModeSelector
from Display import Display
from EnumModeSelector import UploadMode
from EnumModeSelector import MaskingOption
from VideoHelper import VideoHelper
import os
import tempfile
from mtcnn.mtcnn import MTCNN
from MaskingModeHelper import MaskingModeHelper
import dlib


class Application:
    """
    Application class creates the main interface and is responsible for the program flow.
    """
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
    recognition_result = None
    person_name = None
    face_mesh_mode = None
    video = None
    detector = None
    print_allert_face_detected = None
    interpolation_mode = None
    dlib_detector = None

    def __init__(self):
        Information.print_page_title()

    def load_interface(self):
        program_mode = ModeSelector.load_program_mode()
        self.detector = MTCNN()
        self.dlib_detector = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

        if program_mode == UploadMode.upload_default:
            Information.print_main_page()

        if program_mode == UploadMode.upload_image:
            self.load_image_mode()
            if self.image_loaded:
                Display.load_image_on_sidebar(self.image_loaded)

        if program_mode == UploadMode.upload_video:
            self.load_video_mode()

        if program_mode == UploadMode.upload_live_camera:
            self.run_application_camera()

        if self.image_loaded:
            self.run_application_image()

        if self.video_loaded:
            self.run_application_video()

    def load_image_mode(self):
        image_loaded = st.sidebar.file_uploader(
            "Please select image to upload", type=["png", "jpg", "jpeg"], key="upload_1"
        )
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_loaded = image
            self.prepare_image_to_draw_on()
            self.print_allert_face_detected = True

    def load_video_mode(self):
        video_loaded = st.sidebar.file_uploader(
            "Please select image to upload",
            type=["mp4", "mov", "avi"],
            key="upload_video",
        )

        if video_loaded:
            video = tempfile.NamedTemporaryFile(delete=False)
            video.write(video_loaded.read())
            self.video_loaded = video
            self.print_allert_face_detected = False

    def load_image_to_learn_recognition_mode(self):
        image_loaded = st.sidebar.file_uploader(
            "Please select image to upload", type=["png", "jpg", "jpeg"], key="upload_2"
        )
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_learn_recognition_loaded = image

    def load_detection_mode(self):
        ((self.detection_mode),
         (self.detection_mode_colors),
         (self.detection_mode_sizes),
         self.face_mesh_mode,
         ) = ModeSelector.load_detection_mode()

    def load_masking_mode(self):
        mask_mode, size = ModeSelector.load_mask_mode()
        self.masking_mode = mask_mode
        self.masking_size = size

    def enable_face_detection(self):
        is_face_detection_enabled = ModeSelector.load_face_detector_check_box()
        self.is_face_detection_enabled = is_face_detection_enabled

    def enable_face_recognition(self):
        is_face_recognition_enabled = ModeSelector.load_face_recognition_check_box()
        if is_face_recognition_enabled:
            self.box_on_faces = True

        self.is_face_recognition_enabled = is_face_recognition_enabled

    def prepare_image_to_draw_on(self):
        image = cv2.imread(self.image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_after_masking = image_rgb

    def draw_on_image(self):
        if self.box_on_faces:
            Display.draw_box_on_faces(
                self.faces,
                self.image_after_masking,
                (255, 0, 0),
                self.print_allert_face_detected,
            )

            self.image_after_masking = Display.write_person_name_on_face(
                self.recognition_result,
                self.faces,
                self.image_after_masking,
                self.person_name,
            )

        self.image_after_masking = Display.draw_face_features(
            self.detection_mode,
            self.image_after_masking,
            self.faces,
            self.detection_mode_colors,
            self.detection_mode_sizes,
            self.face_mesh_mode,
            self.dlib_detector
        )

    def load_all_checkboxes(self):
        self.enable_face_detection()
        if self.is_face_detection_enabled:
            self.box_on_faces = ModeSelector.load_box_on_face_check_box()
            self.load_detection_mode()
            self.load_masking_mode()

            if self.masking_mode == MaskingOption.extract_face_features_interpolation:
                self.interpolation_mode = ModeSelector.load_interpolation_mode()

        self.enable_face_recognition()
        if self.is_face_recognition_enabled:
            self.load_image_to_learn_recognition_mode()
            if self.image_learn_recognition_loaded:
                Display.load_image_on_sidebar(self.image_learn_recognition_loaded)
                self.person_name = Display.get_person_name_label()
                self.recognition_mode = ModeSelector.load_recognition_mode_check_box()

    def run_application(self, frame):
        self.image_after_masking = frame
        self.image_after_masking = MaskingModeHelper.run_masking_mode(
            self.masking_mode,
            self.image_after_masking,
            self.masking_size,
            self.interpolation_mode,
        )
        self.faces = FaceDetector.detect_faces(self.image_after_masking, self.detector)
        if self.is_face_recognition_enabled:
            self.recognition_result = FaceRecognizer.recognize_faces(
                self.image_after_masking,
                self.image_learn_recognition_loaded,
                self.recognition_mode,
            )
        self.draw_on_image()

    def run_application_image(self):
        self.load_all_checkboxes()
        if self.is_face_detection_enabled:
            self.run_application(self.image_after_masking)
        Display.view_image(self.image_after_masking)
        self.save_image(self.image_after_masking)

    def run_application_video(self):
        self.load_all_checkboxes()
        if self.is_face_detection_enabled and st.sidebar.button("play"):
            frames = Display.load_video(self.video_loaded)
            images_after_masking = []
            my_bar = st.progress(0)
            n_frames = len(frames)
            percentage_per_frame = 100 / n_frames

            for index, frame in enumerate(frames):
                my_bar.progress(int(index * percentage_per_frame))
                self.run_application(frame)
                print(id(self.image_after_masking))
                images_after_masking.append(self.image_after_masking)
                gc.collect()
            VideoHelper.save_video(images_after_masking)
            VideoHelper.display_video()

    def run_application_camera(self):
        self.load_all_checkboxes()
        if self.is_face_detection_enabled and st.sidebar.button("play"):
            cam = cv2.VideoCapture(0)
            st_frame = st.empty()
            index = 0
            while True:
                ret, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.run_application(frame)
                gc.collect()
                st_frame.image(self.image_after_masking, use_column_width=True)
                self.save_camera(index, self.image_after_masking)
                index = index+1

    def save_image(self, images_after_masking):
        try:
            os.mkdir(".\\image")
        except Exception:
            print('something went wrong')
        image_rgb = cv2.cvtColor(images_after_masking, cv2.COLOR_BGR2RGB)
        cv2.imwrite("image\\img.jpg", image_rgb)

    def save_camera(self, index, images_after_masking):
        try:
            os.mkdir(".\\camera")
        except Exception:
            print('something went wrong')
        image_rgb = cv2.cvtColor(images_after_masking, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"camera\\img_{str(index)}.jpg", image_rgb)


