from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN


class FaceDetector:

    @staticmethod
    def detect_faces(img, img2=None):
        #Use masked image(img2) instead original
        #pixels = pyplot.imread(img)
        decetor = MTCNN()
        faces = decetor.detect_faces(img2)
        return faces

    def face_detection_mtcnn(self, img, face_distance=None, person_name=None, display=True):
        pixels = pyplot.imread(img)
        decetor = MTCNN()
        faces = decetor.detect_faces(pixels)
        self.draw_image_with_boxes(img, faces, face_distance, person_name, display)

        #  if display:
        #      st.write("Face was detected")
