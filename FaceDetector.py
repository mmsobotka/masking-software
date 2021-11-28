import dlib


class FaceDetector:
    @staticmethod
    def detect_faces(img, detector=None):
        # TODO detector mtcnn takes a lot of time
        faces = detector.detect_faces(img)
        return faces

    @staticmethod
    def detect_faces_dlib(image):
        face_detector = dlib.get_frontal_face_detector()
        return face_detector(image, 1)
