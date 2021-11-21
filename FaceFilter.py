import cv2
import mediapipe as mp
import numpy as np


class FaceFilter:
    right_eye_indices = [342, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 261, 446]
    left_eye_indices = [189, 244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221]
    mouth_indices = [37, 0, 267, 269, 410, 287, 273, 405, 314, 17, 84, 181, 43, 57, 132, 39]
    nose_indices = [413, 465, 412, 399, 456, 363, 440, 457, 458, 354, 19, 125, 238, 44, 45, 236, 174, 188, 122]
    face_without_forehead_chin_indices = [70, 53, 52, 55, 8, 285, 295, 282, 283, 276, 353, 264, 447, 345, 447, 352, 376, 433, 435, 416, 367, 430, 431, 418, 421, 200, 194, 204, 202, 138, 214, 192, 213, 187, 123, 116, 143, 156]

    @staticmethod
    def run_face_gausian_filter(faces, image):
        for face in faces:
            x, y, width, height = face['box']
            face = image[y:y + height, x:x + width]
            blur = cv2.blur(face, (42, 42))
            image[y:y + height, x:x + width] = blur


    @staticmethod
    def cp(image_loaded, image_to_draw_on):
        image = cv2.imread(image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)
        results = face_mesh.process(image_rgb)

        height, width, _ = image_to_draw_on.shape

        print(image_to_draw_on.shape)
        right_eye = []

        img_mark = image_to_draw_on.copy()
        mark = np.zeros(image_to_draw_on.shape[:2], np.uint8)

        if results.multi_face_landmarks:
            print(len(results.multi_face_landmarks))
            indices = FaceFilter.right_eye_indices
            for index in indices:
                x = results.multi_face_landmarks[0].landmark[index].x
                y = results.multi_face_landmarks[0].landmark[index].y
                x = int(x * width)
                y = int(y * height)
                right_eye.append([x, y])
            cv2.fillPoly(mark, [np.array([right_eye], np.int32)], (255, 0, 0))

            image_to_draw_on = cv2.inpaint(img_mark, mark, 6, cv2.INPAINT_TELEA)

        return image_to_draw_on
