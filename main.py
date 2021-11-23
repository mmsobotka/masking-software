import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import tempfile
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import face_recognition
import pyautogui

from Application import *


class MyClass:
    UNKNOWN_IMAGE = None
    KNOWN_IMAGE = None
    PERSON_NAME = None
    PERSON_ENCODING = None
    FACE_DISTANCE = None
    RECOGNITION_MODE = None
    POINTS_SELECTED = False
    LINES_SELECTED = False
    MASK = False
    FACE_IMAGE_MASKED = None

    def print_main_page(self):
        notes = f"""

        🔍 ** Information **
        - Face biometric masking software

        👋 ** Help **
        - On the left side of the screen there is a section with options to choose from
        - Select the object type for analysis - Image, Video or Live Camera view
        - Next for Image or Video options load the selected file from the folder, for Live Camera option click "Run" button
        - Choose how you want to display facial feature points
        - Press the "Face detection" button to detect a face in the image
        - Using the "Save plot with face detection" button you can save the plot with the obtained face detection on the image
        - In order to use the face recognition functionality on an image already loaded on the page, open another file from the folder, which represents a photo of a person on the basis of which you want to use face recognition
        - Select HOG or CNN method
        - Enter the name of the person to be recognized
        - Press the "Face recognition" button
        - To refresh the page press the "Reset" button
        """
        st.write(notes)


    def set_up_gui(self):
        # st.set_page_config(layout="wide")
        # st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
        #           unsafe_allow_html=True)
        """---"""

        UPLOAD_DEFAULT = "Mode Selection"
        UPLOAD_IMAGE = "Load Image"
        UPLOAD_VIDEO = "Load Video"
        LIVE_CAMERA = "Live Camera"

        SIDEBAR_OPTIONS = [UPLOAD_DEFAULT, UPLOAD_IMAGE, UPLOAD_VIDEO, LIVE_CAMERA]
        program_mode = st.sidebar.selectbox('Choose a target for analysis', SIDEBAR_OPTIONS)

        if program_mode == UPLOAD_DEFAULT:
            self.print_main_page()

        # done 1

        if program_mode == UPLOAD_IMAGE:
            uploaded_img = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                    key="upload_1")
            if uploaded_img is not None:
                temp_img = tempfile.NamedTemporaryFile(delete=False)
                temp_img.write(uploaded_img.read())
                self.UNKNOWN_IMAGE = temp_img
                self.display_image(temp_img)
                self.set_landmrks()

        if program_mode == UPLOAD_VIDEO:
            uploaded_vid = st.sidebar.file_uploader("Please select video to upload", type=['mp4', 'mov'])

            if uploaded_vid is not None:
                temp_vid = tempfile.NamedTemporaryFile(delete=False)
                temp_vid.write(uploaded_vid.read())
                self.display_video(temp_vid)
                self.set_landmrks()

        if program_mode == LIVE_CAMERA:
            st.title("Live Camera")
            run_camera = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            cam = cv2.VideoCapture(1)

            while run_camera:
                ret, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

        mask = st.sidebar.checkbox('Mask')
        if mask:
            self.MASK = True

        if st.sidebar.button('Face detection'):
            self.face_detection_mtcnn(self.UNKNOWN_IMAGE)
        else:
            pass

        learn_rec = st.sidebar.file_uploader("Please select image to learn recognition", type=['png', 'jpg', 'jpeg'],
                                             key="upload_2")
        if learn_rec is not None:
            temp_rec = tempfile.NamedTemporaryFile(delete=False)
            temp_rec.write(learn_rec.read())
            self.KNOWN_IMAGE = temp_rec
            self.display_image(temp_rec)

        mode_cnn = st.sidebar.checkbox('CNN')
        if mode_cnn:
            st.write('You selected CNN.')
            self.RECOGNITION_MODE = "cnn"
        else:
            pass

        mode_hog = st.sidebar.checkbox('HOG')
        if mode_hog:
            st.write('You selected HOG.')
            self.RECOGNITION_MODE = "hog"
        else:
            pass

        caption = st.sidebar.text_input('Person name', ' ')
        self.PERSON_NAME = caption
        # st.sidebar.write(' ', caption)

        if st.sidebar.button('Face recognition'):
            self.face_detection_mtcnn(self.UNKNOWN_IMAGE, display=False)
            self.face_recognition_function(self.UNKNOWN_IMAGE, self.KNOWN_IMAGE)
        else:
            pass

        if st.sidebar.button('Save plot with face detection'):
            pixels = pyplot.imread(self.UNKNOWN_IMAGE)
            decetor = MTCNN()
            faces = decetor.detect_faces(pixels)
            self.draw_plot_with_boxes(self.UNKNOWN_IMAGE, faces)
        else:
            pass

    def set_landmrks(self):
        points = st.sidebar.checkbox('Landmarks as points')
        if points:
            self.POINTS_SELECTED = True

        lines = st.sidebar.checkbox('Landmarks as lines')
        if lines:
            self.LINES_SELECTED = True

    def reload_image(self, img):
        #image_location = st.empty()
        #image_location.image(img)
        st.image(img, use_column_width=True)
        # col1, col2, col3 = st.columns([0.5, 1, 0.5])
        # col2.image(img, use_column_width=False)

    def display_image(self, img):
        image = Image.open(img)
        # st.image(image, caption='Load')
        # reload_image(image)
        st.sidebar.image(image)

    def display_video(self, vid):
        vf = cv2.VideoCapture(vid.name)
        stframe = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                print("Can't receive frame - stream end.")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb)

    def simple_mask(self, img, box):
        x, y, width, height = box
        face = img[y:y + height, x:x + width]
        blur = cv2.blur(face, (42, 42))
        img[y:y + height, x:x + width] = blur



    def detect_eyes(self, img):
        face_landmark_list = face_recognition.face_landmarks(img)
        for face_landmarks in face_landmark_list:
            print(face_landmarks)
            right_eye = []
            left_eye = []

            for i in range(6):
                point_re = list(face_landmarks["right_eye"][i])
                print(point_re)
                right_eye.append(point_re)
                point_le = list(face_landmarks['left_eye'][i])
                left_eye.append(point_le)

            print(right_eye)

            cv2.fillPoly(img, [np.array([right_eye], np.int32)], (0, 0, 0))
            cv2.fillPoly(img, [np.array([left_eye], np.int32)], (0, 0, 0))

    def detect_lips(self, img):
        face_landmark_list = face_recognition.face_landmarks(img)
        print(face_landmark_list)
        for face_landmarks in face_landmark_list:
            tlip = []
            blip = []
            for i in range(12):
                point_re = list(face_landmarks["top_lip"][i])
                tlip.append(point_re)
                point_le = list(face_landmarks['bottom_lip'][i])
                blip.append(point_le)

            cv2.fillPoly(img, [np.array([tlip], np.int32)], (0, 0, 0))
            cv2.fillPoly(img, [np.array([blip], np.int32)], (0, 0, 0))

    def face_detection_mtcnn(self, img, face_distance=None, person_name=None, display=True):
        pixels = pyplot.imread(img)
        decetor = MTCNN()
        faces = decetor.detect_faces(pixels)
        self.draw_image_with_boxes(img, faces, face_distance, person_name, display)
        if display:
            st.write("Face was detected")

    def draw_image_with_boxes(self, filename, faces, face_distance=None, person_name=None, display=True):
        img = cv2.imread(filename.name)
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for face in faces:
            x, y, width, height = face['box']

            if self.MASK:
                #self.simple_mask(imageRGB, face['box'])
                self.detect_eyes(imageRGB)
                #self.detect_lips(imageRGB)
                self.FACE_IMAGE_MASKED = imageRGB.copy()

            cv2.rectangle(imageRGB, (x, y), (x + width, y + height), (255, 0, 0), 2)

            if face_distance is not None:
                cv2.rectangle(imageRGB, (x, y + height), (x + width, y + height + 35), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX

                face_distance = 1.0 - face_distance
                face_distance *= 100
                face_distance = int(face_distance)

                text_to_draw = person_name + " " + str(face_distance) + "%"

                if face_distance < 40:
                    person_name = "UNKNOWN"
                    text_to_draw = person_name

                cv2.putText(imageRGB, text_to_draw, (x + 6, y + height + 28), font, 0.8, (255, 255, 255), 1)
            else:
                if self.POINTS_SELECTED:
                    for key, value in face['keypoints'].items():
                        cv2.circle(imageRGB, value, 2, (255, 0, 0), 3)
                if self.LINES_SELECTED:
                    face_landmarks_list = face_recognition.face_landmarks(imageRGB)
                    pil_image = Image.fromarray(imageRGB)
                    for face_landmarks in face_landmarks_list:
                        d = ImageDraw.Draw(pil_image, 'RGBA')
                        #d.line(face_landmarks['chin'], width=2)
                        #d.line(face_landmarks['left_eyebrow'], width=2)
                        #d.line(face_landmarks['right_eyebrow'], width=2)
                        #d.line(face_landmarks['nose_bridge'], width=2)
                        #d.line(face_landmarks['nose_tip'], width=2)
                        #d.line(face_landmarks['left_eye'], width=2)
                        #d.line(face_landmarks['right_eye'], width=2)

                        #d.line(face_landmarks['top_lip'], width=2)
                        #d.line(face_landmarks['bottom_lip'], width=2)

                        # Make the eyebrows into a nightmare
                        d.polygon(face_landmarks['nose_tip'], fill=(255, 255, 255))
                        #d.polygon(face_landmarks['nose_bridge'], fill=(0, 0, 0))
                        #d.line(face_landmarks['nose_bridge'], fill=(0, 0, 0), width=4)
                        d.line(face_landmarks['nose_tip'], fill=(255, 255, 255), width=4)

                        # Gloss the lips
                        d.polygon(face_landmarks['top_lip'], fill=(255, 255, 255))
                        d.polygon(face_landmarks['bottom_lip'], fill=(255, 255, 255))
                        d.line(face_landmarks['top_lip'], fill=(255, 255, 255), width=5)
                        d.line(face_landmarks['bottom_lip'], fill=(255, 255, 255), width=4)

                        # Sparkle the eyes
                        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255))
                        d.line(face_landmarks['left_eye'], fill=(255, 255, 255), width=10)
                        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255))
                        d.line(face_landmarks['right_eye'], fill=(255, 255, 255), width=10)

                        # Apply some eyeliner
                        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(255, 255, 255),
                               width=15)
                        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(255, 255, 255),
                               width=15)

                    imageRGB = np.array(pil_image)

        if display:
            self.reload_image(imageRGB)

    def draw_plot_with_boxes(self, filename, result_list):
        data = pyplot.imread(filename)
        pyplot.imshow(data)
        ax = pyplot.gca()
        for result in result_list:
            x, y, width, height = result['box']
            box = Rectangle((x, y), width, height, fill=False, color='red')
            ax.add_patch(box)
            for key, value in result['keypoints'].items():
                dot = Circle(value, radius=2, color='red')
                ax.add_patch(dot)
        pyplot.savefig("image.png")

    def face_recognition_function(self, UNKNOWN_IMAGE, KNOWN_IMAGE):
        recognition_mode = "cnn"
        if self.RECOGNITION_MODE is not None:
            recognition_mode = self.RECOGNITION_MODE

        known_person = face_recognition.load_image_file(KNOWN_IMAGE)

        unknown_person = face_recognition.load_image_file(UNKNOWN_IMAGE)
        if self.FACE_IMAGE_MASKED is not None:
            print('log')
            unknown_person = self.FACE_IMAGE_MASKED

        PERSON_ENCODING = face_recognition.face_encodings(known_person, model=recognition_mode)[0]
        unknown_person_encoding = face_recognition.face_encodings(unknown_person, model=recognition_mode)
        if len(unknown_person_encoding) == 0:
            self.FACE_DISTANCE = 1.0
            self.face_detection_mtcnn(UNKNOWN_IMAGE, self.FACE_DISTANCE, self.PERSON_NAME)
        else:
            unknown_person_encoding = unknown_person_encoding[0]
            print(unknown_person_encoding)
            # results = face_recognition.compare_faces([PERSON_ENCODING], unknown_person_encoding)
            self.FACE_DISTANCE = face_recognition.face_distance([PERSON_ENCODING], unknown_person_encoding)
            self.face_detection_mtcnn(UNKNOWN_IMAGE, self.FACE_DISTANCE, self.PERSON_NAME)


if __name__ == '__main__':
    #st.set_page_config(layout="wide")
    #st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
    #            unsafe_allow_html=True)
    #"""---"""
    #application = MyClass()
    #application.set_up_gui()

    #"""---"""
    #if st.button("Reset"):
    #    pyautogui.hotkey("ctrl", "F5")

    application = Application()
    """---"""
    application.load_interface()

