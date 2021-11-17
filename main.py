from PIL import Image
import streamlit as st
import tempfile
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import face_recognition

UNKNOWN_IMAGE = None
KNOWN_IMAGE = None
PERSON_NAME = None
PERSON_ENCODING = None
FACE_DISTANCE = None
RECOGNITION_MODE = None


def set_up_gui():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
                unsafe_allow_html=True)
    """---"""

    UPLOAD_DEFAULT = "Mode Selection"
    UPLOAD_IMAGE = "Load Image"
    UPLOAD_VIDEO = "Load Video"
    LIVE_CAMERA = "Live Camera"

    SIDEBAR_OPTIONS = [UPLOAD_DEFAULT, UPLOAD_IMAGE, UPLOAD_VIDEO, LIVE_CAMERA]
    program_mode = st.sidebar.selectbox('Choose a target for analysis', SIDEBAR_OPTIONS)

    if program_mode == UPLOAD_IMAGE:
        uploaded_img = st.sidebar.file_uploader("Please select image to upload", type=['png', 'jpg', 'jpeg'],
                                                key="upload_1")
        if uploaded_img is not None:
            temp_img = tempfile.NamedTemporaryFile(delete=False)
            temp_img.write(uploaded_img.read())
            UNKNOWN_IMAGE = temp_img
            display_image(temp_img)

    if program_mode == UPLOAD_VIDEO:
        uploaded_vid = st.sidebar.file_uploader("Please select video to upload", type=['mp4', 'mov'])

        if uploaded_vid is not None:
            temp_vid = tempfile.NamedTemporaryFile(delete=False)
            temp_vid.write(uploaded_vid.read())
            display_video(temp_vid)

    if program_mode == LIVE_CAMERA:
        st.title("Live Camera")
        run_camera = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        cam = cv2.VideoCapture(1)

        while run_camera:
            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

    if st.sidebar.button('Face detection'):
        face_detection_mtcnn(UNKNOWN_IMAGE)
    else:
        pass

    learn_rec = st.sidebar.file_uploader("Please select image to learn recognition", type=['png', 'jpg', 'jpeg'],
                                         key="upload_2")
    if learn_rec is not None:
        temp_rec = tempfile.NamedTemporaryFile(delete=False)
        temp_rec.write(learn_rec.read())
        KNOWN_IMAGE = temp_rec
        display_image(temp_rec)

    global RECOGNITION_MODE

    mode_cnn = st.sidebar.checkbox('CNN')
    if mode_cnn:
        st.write('You selected CNN.')
        RECOGNITION_MODE = "cnn"
    else:
        pass

    mode_hog = st.sidebar.checkbox('HOG')
    if mode_hog:
        st.write('You selected HOG.')
        RECOGNITION_MODE = "hog"
    else:
        pass

    caption = st.sidebar.text_input('Person name', ' ')
    global PERSON_NAME
    PERSON_NAME = caption
    # st.sidebar.write(' ', caption)

    if st.sidebar.button('Face recognition'):
        face_recognition_function(UNKNOWN_IMAGE, KNOWN_IMAGE)
    else:
        pass

    if st.sidebar.button('save plot with face detection'):
        pixels = pyplot.imread(UNKNOWN_IMAGE)
        decetor = MTCNN()
        faces = decetor.detect_faces(pixels)
        draw_plot_with_boxes(UNKNOWN_IMAGE, faces)
    else:
        pass


def reload_image(img):
    image_location = st.empty()
    image_location.image(img)
    # col1, col2, col3 = st.columns([0.5, 1, 0.5])
    # col2.image(img, use_column_width=False)


def display_image(img):
    image = Image.open(img)
    # st.image(image, caption='Load')
    # reload_image(image)
    st.sidebar.image(image)


def display_video(vid):
    vf = cv2.VideoCapture(vid.name)
    stframe = st.empty()

    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            print("Can't receive frame - stream end.")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb)


def face_detection_mtcnn(img, face_distance=None, person_name=None):
    pixels = pyplot.imread(img)
    decetor = MTCNN()
    faces = decetor.detect_faces(pixels)
    draw_image_with_boxes(img, faces, face_distance, person_name)


def draw_image_with_boxes(filename, faces, face_distance=None, person_name=None):
    img = cv2.imread(filename.name)
    imageRGB = cv2.cv2tColor(img, cv2.COLOR_BGR2RGB)
    for face in faces:
        x, y, width, height = face['box']
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
            for key, value in face['keypoints'].items():
                cv2.circle(imageRGB, value, 2, (255, 0, 0), 3)

    reload_image(imageRGB)


def draw_plot_with_boxes(filename, result_list):
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


def face_recognition_function(UNKNOWN_IMAGE, KNOWN_IMAGE):
    global RECOGNITION_MODE
    recognition_mode = "cnn"
    if RECOGNITION_MODE is not None:
        recognition_mode = RECOGNITION_MODE

    known_person = face_recognition.load_image_file(KNOWN_IMAGE)
    unknown_person = face_recognition.load_image_file(UNKNOWN_IMAGE)
    PERSON_ENCODING = face_recognition.face_encodings(known_person, model=recognition_mode)[0]
    unknown_person_encoding = face_recognition.face_encodings(unknown_person, model=recognition_mode)
    if len(unknown_person_encoding) == 0:
        FACE_DISTANCE = 1.0
        face_detection_mtcnn(UNKNOWN_IMAGE, FACE_DISTANCE, PERSON_NAME)
    else:
        unknown_person_encoding = unknown_person_encoding[0]
        print(unknown_person_encoding)
        # results = face_recognition.compare_faces([PERSON_ENCODING], unknown_person_encoding)
        FACE_DISTANCE = face_recognition.face_distance([PERSON_ENCODING], unknown_person_encoding)
        face_detection_mtcnn(UNKNOWN_IMAGE, FACE_DISTANCE, PERSON_NAME)


if __name__ == '__main__':
    set_up_gui()
