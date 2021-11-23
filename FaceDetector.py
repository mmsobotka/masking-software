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


