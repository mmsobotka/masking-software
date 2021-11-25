from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from Application import *
import gc


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

    def set_up_gui(self):
            st.title("Live Camera")
            run_camera = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            cam = cv2.VideoCapture(1)

            while run_camera:
                ret, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

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


if __name__ == '__main__':
    #if st.button("Reset"):
    #    pyautogui.hotkey("ctrl", "F5")

    application = Application()
    """---"""
    application.load_interface()
    gc.collect()

