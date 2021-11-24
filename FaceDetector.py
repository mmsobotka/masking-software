from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import datetime

class FaceDetector:

    @staticmethod
    def detect_faces(img, img2=None, detector=None):
        #Use masked image(img2) instead original
        #pixels = pyplot.imread(img)
        #TODO detector mtcnn takes a lot of time
        faces = detector.detect_faces(img2)
        return faces


