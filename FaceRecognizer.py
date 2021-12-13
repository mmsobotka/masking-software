import face_recognition
from EnumModeSelector import FaceDetectionMode


class FaceRecognizer:
    """
    FaceRecognizer class is used for face recognition.
    """
    @staticmethod
    def recognize_faces(unknown_image, model_image, mode):
        if mode == FaceDetectionMode.cnn:
            return FaceRecognizer.face_recognition_function(
                unknown_image, model_image, mode
            )
        elif mode == FaceDetectionMode.hog:
            return FaceRecognizer.face_recognition_function(
                unknown_image, model_image, mode
            )

    @staticmethod
    def face_recognition_function(unknown_image, model_image, mode):
        model_person = face_recognition.load_image_file(model_image)
        unknown_person = unknown_image

        encoding_model_person = face_recognition.face_encodings(
            model_person, model=mode
        )
        if len(encoding_model_person) == 0:
            return 0
        encoding_unknown_person = face_recognition.face_encodings(
            unknown_person, model=mode
        )
        if len(encoding_unknown_person) == 0:
            return 0

        recognized_distance = face_recognition.face_distance(
            encoding_model_person, encoding_unknown_person[0]
        )

        if len(recognized_distance) > 1:
            recognized_distance = recognized_distance[0]

        recognized_distance = int((1 - recognized_distance) * 100)
        return recognized_distance
