import cv2.cv2
import gc
import FaceFilter
from Information import *
from FaceDetector import *
from FaceFilter import *
from FaceRecognizer import *
import tempfile
from mtcnn.mtcnn import MTCNN


# TODO
# restart button
# rescaling of input images
# size of rectangle under box and name text with face and text should be linked with size of face size
# when few faces recognition return list of faces or find the most recognized one
# when face in picture is too small then print warning to upload another image ( check media pipe face sizes internal options)
# check if hog and cnn recognition mode works correctly
# p2 in cut face features user should chose what to cut
# reload with change main mode

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
    recognition_result = None
    person_name = None
    face_mesh_mode = None
    video = None
    detector = None
    print_allert_face_detected = None
    interpolation_mode = None

    def __init__(self):
        Information.print_page_title()

    def load_interface(self):
        program_mode = ModeSelector.load_program_mode()
        self.detector = MTCNN()

        if program_mode == ModeSelector.upload_default:
            Information.print_main_page()

        if program_mode == ModeSelector.upload_image:
            self.load_image_mode()
            if self.image_loaded:
                Display.load_image_on_sidebar(self.image_loaded)

        if program_mode == ModeSelector.upload_video:
            self.load_video_mode()

        if program_mode == ModeSelector.upload_live_camera:
            self.run_application_camera()

        if self.image_loaded:
            self.run_application_image()

        if self.video_loaded:
            self.run_application_video()


    def load_image_mode(self):
        image_loaded = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_1")
        if image_loaded:
            print("image loaded!")
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_loaded = image
            self.prepare_image_to_draw_on()
            self.print_allert_face_detected = True

    def load_video_mode(self):
        video_loaded = st.sidebar.file_uploader("Please select image to upload", type=['mp4', 'mov', 'avi'],
                                                key="upload_video")

        if video_loaded:
            video = tempfile.NamedTemporaryFile(delete=False)
            video.write(video_loaded.read())
            self.video_loaded = video
            self.print_allert_face_detected = False

    def load_image_to_learn_recognition_mode(self):
        image_loaded = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_2")
        if image_loaded:
            image = tempfile.NamedTemporaryFile(delete=False)
            image.write(image_loaded.read())
            self.image_learn_recognition_loaded = image

    def load_detection_mode(self):
        (self.detection_mode), (self.detection_mode_colors), (
            self.detection_mode_sizes), self.face_mesh_mode = ModeSelector.load_detection_mode()

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

    def detect_faces(self):
        self.faces = FaceDetector.detect_faces(self.image_after_masking, self.detector)

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
            Display.draw_box_on_faces(self.faces, self.image_after_masking, (255, 0, 0),
                                      self.print_allert_face_detected)
            self.write_person_name_on_face()
        self.image_after_masking = Display.draw_face_features(self.detection_mode, self.image_after_masking,
                                                              self.faces,
                                                              self.detection_mode_colors,
                                                              self.detection_mode_sizes,
                                                              self.face_mesh_mode)

    def write_person_name_on_face(self):
        if self.recognition_result:
            name = "UNKNOWN"
            if self.recognition_result > 40:
                name = self.person_name + " " + str(self.recognition_result) + "%"
            Display.draw_rectangle_under_faces(self.faces, self.image_after_masking, (255, 0, 0))
            Display.write_names_under_faces(self.faces, self.image_after_masking, (255, 255, 255), name)

    def run_masking_mode(self):
        if self.masking_mode == ModeSelector.default:
            pass
        elif self.masking_mode == ModeSelector.gaussian_filter:
            # FaceFilter.run_face_gausian_filter(self.faces, self.image_after_masking)
            self.image_after_masking = FaceFilter.run_face_gausian_filter(self.image_after_masking,
                                                                          self.masking_size,
                                                                          FaceFilter.face_without_forehead_chin_indices)

        elif self.masking_mode == ModeSelector.extract_face_features:
            self.image_after_masking = FaceFilter.run_face_cut_features(self.image_after_masking,
                                                                        FaceFilter.left_eye_indices)
            self.image_after_masking = FaceFilter.run_face_cut_features(self.image_after_masking,
                                                                        FaceFilter.right_eye_indices)
        elif self.masking_mode == ModeSelector.accurate_extract_face_features:
            self.image_after_masking = FaceFilter.run_face_cut_features2(self.image_after_masking)

        #TODO
        # elif self.masking_mode == ModeSelector.face_transform:
        #    pass
        elif self.masking_mode == ModeSelector.extract_face_features_interpolation:
            self.run_interpolation_mode(self.interpolation_mode)
            # self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
            #   self.image_loaded, self.image_after_masking, )

    def run_interpolation_mode(self, interpolation_mode):
        right_eye, left_eye, nose, mouth = interpolation_mode
        if right_eye:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_after_masking,
                self.masking_size,
                FaceFilter.right_eye_indices)
        if left_eye:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_after_masking,
                self.masking_size,
                FaceFilter.left_eye_indices)
        if nose:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_after_masking,
                self.masking_size,
                FaceFilter.nose_indices)
        if mouth:
            self.image_after_masking = FaceFilter.run_face_filter_face_features_extraction_interpolation(
                self.image_after_masking,
                self.masking_size,
                FaceFilter.mouth_indices)

    def get_person_name(self):
        self.person_name = Display.get_person_name_label()

    def run_application_image(self):
        self.enable_face_detection()
        if self.is_face_detection_enabled:
            self.load_box_on_face_check_box()  # checkbox only
            self.load_detection_mode()  # checkbox only
            self.load_masking_mode()  # checkbox only

            if self.masking_mode == ModeSelector.extract_face_features_interpolation:
                self.interpolation_mode = ModeSelector.load_interpolation_mode()  # todo change name

            self.run_masking_mode()
            self.detect_faces()

        self.enable_face_recognition()  # checkbox only
        if self.is_face_recognition_enabled:
            self.load_image_to_learn_recognition_mode()
            if self.image_learn_recognition_loaded:
                Display.load_image_on_sidebar(self.image_learn_recognition_loaded)
                self.get_person_name()
                self.load_recognition_mode_check_box()
                self.recognition_result = FaceRecognizer.recognize_faces(self.image_after_masking,
                                                                         self.image_learn_recognition_loaded,
                                                                         self.recognition_mode)
                # st.write(self.recognition_result)

        if self.is_face_detection_enabled:
            self.draw_on_image()

        Display.view_image(self.image_after_masking)

    def run_application_video(self):
        self.enable_face_detection()  # checkbox only
        if self.is_face_detection_enabled:
            self.load_box_on_face_check_box()  # checkbox only
            self.load_detection_mode()  # checkbox only
            self.load_masking_mode()  # checkbox only

            if self.masking_mode == ModeSelector.extract_face_features_interpolation:
                self.interpolation_mode = ModeSelector.load_interpolation_mode()  # todo change name

        self.enable_face_recognition()
        if self.is_face_recognition_enabled:
            self.load_image_to_learn_recognition_mode()  # file uploader only
            if self.image_learn_recognition_loaded:
                Display.load_image_on_sidebar(self.image_learn_recognition_loaded)  # ?
                self.get_person_name()  # once
                self.load_recognition_mode_check_box()  # checkbox only
                # st.write(self.recognition_result)

        if self.is_face_detection_enabled and st.sidebar.button("play"):
            frames = Display.load_video(self.video_loaded)
            images_after_masking = []
            my_bar = st.progress(0)
            n_frames = len(frames)
            percentage_per_frame = 100 / n_frames

            for index, frame in enumerate(frames):
                my_bar.progress(int(index * percentage_per_frame))
                self.image_after_masking = frame
                self.run_masking_mode()
                self.detect_faces()
                self.draw_on_image()  # per frame

                if self.is_face_recognition_enabled:
                    self.recognition_result = FaceRecognizer.recognize_faces(self.image_after_masking,
                                                                             self.image_learn_recognition_loaded,
                                                                             self.recognition_mode)  # per frame

                images_after_masking.append(self.image_after_masking)
                del self.image_after_masking  #
                gc.collect()

            del frames
            del self.faces
            del self.detector

            gc.collect()

            self.save_video(images_after_masking)
            self.load_video()

    def run_application_camera(self):
        print("run1")
        self.enable_face_detection()  # checkbox only
        if self.is_face_detection_enabled:
            self.load_box_on_face_check_box()  # checkbox only
            self.load_detection_mode()  # checkbox only
            self.load_masking_mode()  # checkbox only

            if self.masking_mode == ModeSelector.extract_face_features_interpolation:
                self.interpolation_mode = ModeSelector.load_interpolation_mode()  # todo change name

        self.enable_face_recognition()
        print("run2")
        if self.is_face_recognition_enabled:
            self.load_image_to_learn_recognition_mode()  # file uploader only
            if self.image_learn_recognition_loaded:
                Display.load_image_on_sidebar(self.image_learn_recognition_loaded)  # ?
                self.get_person_name()  # once
                self.load_recognition_mode_check_box()  # checkbox only
                # st.write(self.recognition_result)

        if self.is_face_detection_enabled and st.sidebar.button("play"):
            cam = cv2.VideoCapture(0)

            stframe = st.empty()
            while True:
                ret, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                print(frame.shape)

                self.image_after_masking = frame
                self.run_masking_mode()
                self.detect_faces()
                self.draw_on_image()  # per frame

                if self.is_face_recognition_enabled:
                    self.recognition_result = FaceRecognizer.recognize_faces(self.image_after_masking,
                                                                             self.image_learn_recognition_loaded,
                                                                             self.recognition_mode)  # per frame

                gc.collect()


                stframe.image(self.image_after_masking)


    def save_video(self, images_after_masking):
        for index, image in enumerate(images_after_masking):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('images\\img_' + str(index) + '.jpg', image_rgb)

        size = (images_after_masking[0].shape[1], images_after_masking[0].shape[0])

        out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'avi1'), 30, size)
        for index, image in enumerate(images_after_masking):
            img = cv2.imread('images\\img_' + str(index) + '.jpg')
            out.write(img)
        out.release()

    def load_video(self):
        video_file = open('project.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)




