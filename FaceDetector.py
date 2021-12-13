import dlib


class FaceDetector:
    """
    Class is responsible for finding faces on images.
    Face detector has two methods supported.
    - MTCNN - model based on multitask cascaded convolutional network,
    which includes 3 main parts: "P-net", "R-net", "O-net".
    - dlib frontal face detector - detector based on histogram of oriented gradients combined with a linear SVM classifier,
     an image pyramid, and sliding window detection schema.
    """
    @staticmethod
    def detect_faces(img, detector=None):
        faces = detector.detect_faces(img)
        return faces

    @staticmethod
    def detect_faces_dlib(image):
        face_detector = dlib.get_frontal_face_detector()
        return face_detector(image, 1)
