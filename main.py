from PIL import Image
import streamlit as st
import tempfile
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import cv2 as cv
from matplotlib.patches import Rectangle


INPUT_FILE = None

def set_up_gui():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
                unsafe_allow_html=True)
    """---"""

    UPLOAD_IMAGE = "Upload Image"

    SIDEBAR_OPTIONS = [UPLOAD_IMAGE]
    st.sidebar.header('Mode Selection')
    program_mode = st.sidebar.selectbox('Choose a target for analysis', SIDEBAR_OPTIONS)

    if program_mode == UPLOAD_IMAGE:
        uploaded_file = st.sidebar.file_uploader("Please select to upload image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            temp_img = tempfile.NamedTemporaryFile(delete=False)
            temp_img.write(uploaded_file.read())
            INPUT_FILE = temp_img
            display_image(temp_img)

    if st.sidebar.button('Face detection'):
            face_detection_mtcnn(INPUT_FILE)
    else:
        pass
    if st.sidebar.button('Save plot with face detection'):
            pixels = pyplot.imread(INPUT_FILE)
            decetor = MTCNN()
            faces = decetor.detect_faces(pixels)
            draw_plot_with_boxes(INPUT_FILE, faces)
    else:
        pass

def display_image(img):
    image = Image.open(img)
    st.image(image, caption='Load')
    # st.sidebar.image(image, caption='Load')

def face_detection_mtcnn(img):
    pixels = pyplot.imread(img)
    decetor = MTCNN()
    faces = decetor.detect_faces(pixels)
    draw_image_with_boxes(img, faces)

def draw_image_with_boxes(filename, faces):
    img = cv.imread(filename.name)
    imageRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    for face in faces:
        x, y, width, height = face['box']
        cv.rectangle(imageRGB, (x, y), (x+width, y+height), (255, 0, 0), 2)
    st.image(imageRGB, caption='Load')

def draw_plot_with_boxes(filename, result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        box = Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(box)
    pyplot.savefig("image.png")

if __name__ == '__main__':
    set_up_gui()


