import cv2
import mediapipe as mp
import numpy as np


class FaceFilter:
    right_eye_indices = [342, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 261, 446]
    left_eye_indices = [189, 244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221]
    mouth_indices = [37, 0, 267, 269, 410, 287, 273, 405, 314, 17, 84, 181, 43, 57, 132, 39]
    nose_indices = [413, 465, 412, 399, 456, 363, 440, 457, 458, 354, 19, 125, 238, 44, 45, 236, 174, 188, 122]
    face_without_forehead_chin_indices = [135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367, 288, 361, 323,
                                          454, 356, 389, 251, 284, 298, 333, 299, 337, 151, 108, 69, 104, 68, 21, 162,
                                          127, 234, 93, 58, 138]

    @staticmethod
    def get_mask_polygon(image_loaded, image_to_draw_on, indices):
        image = cv2.imread(image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)
        results = face_mesh.process(image_rgb)

        height, width, _ = image_to_draw_on.shape
        polygon_positions = []
        mask_image = np.zeros(image_to_draw_on.shape[:2], np.uint8)

        if results.multi_face_landmarks:
            for index in indices:
                x = results.multi_face_landmarks[0].landmark[index].x
                y = results.multi_face_landmarks[0].landmark[index].y
                x = int(x * width)
                y = int(y * height)
                polygon_positions.append([x, y])

            cv2.fillPoly(mask_image, [np.array([polygon_positions], np.int32)], (255, 0, 0))

        return mask_image

    @staticmethod
    def run_face_gausian_filter(image_loaded, image_to_draw_on, indices):
        image = image_to_draw_on.copy()
        blur = cv2.blur(image, (42, 42))
        mask = FaceFilter.get_mask_polygon(image_loaded, image_to_draw_on, indices)

        for y, row in enumerate(mask):
            for x, pixel in enumerate(row):
                if pixel:
                    image_to_draw_on[y, x] = blur[y, x]

        return image_to_draw_on

    @staticmethod
    def run_face_filter_face_features_extraction_interpolation(image_loaded, image_to_draw_on, indices):
        mask = FaceFilter.get_mask_polygon(image_loaded, image_to_draw_on, indices)
        image_to_draw_on = cv2.inpaint(image_to_draw_on, mask, 10, cv2.INPAINT_TELEA)
        return image_to_draw_on
