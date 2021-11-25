
class FaceDetector:

    @staticmethod
    def detect_faces(img, detector=None):
        # TODO detector mtcnn takes a lot of time
        faces = detector.detect_faces(img)
        return faces
