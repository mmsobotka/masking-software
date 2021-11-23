import cv2
import mediapipe as mp
import numpy as np
import face_recognition


class FaceFilter:
    right_eye_indices = [342, 445, 444, 443, 442, 441, 413, 464, 453, 452, 451, 450, 449, 448, 261, 446]
    left_eye_indices = [189, 244, 233, 232, 231, 230, 229, 228, 31, 226, 113, 225, 224, 223, 222, 221]
    mouth_indices = [37, 0, 267, 269, 410, 287, 273, 405, 314, 17, 84, 181, 57, 58, 212, 39]
    nose_indices = [413, 465, 412, 399, 456, 363, 440, 457, 458, 354, 19, 125, 238, 44, 45, 236, 174, 188, 122]
    face_without_forehead_chin_indices = [135, 169, 170, 140, 171, 175, 396, 369, 395, 394, 364, 367, 288, 361, 323,
                                          454, 356, 389, 251, 284, 333, 299, 337, 151, 108, 69, 104, 68, 21, 162,
                                          127, 234, 93, 58, 138]

    @staticmethod
    def get_mask_polygon_positions(image_loaded, image_to_draw_on, indices):
        image = cv2.imread(image_loaded.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10)
        results = face_mesh.process(image_rgb)

        height, width, _ = image_to_draw_on.shape
        polygon_positions = []

        if results.multi_face_landmarks:
            for index in indices:
                x = results.multi_face_landmarks[0].landmark[index].x
                y = results.multi_face_landmarks[0].landmark[index].y
                x = int(x * width)
                y = int(y * height)
                polygon_positions.append([x, y])

        return polygon_positions

    @staticmethod
    def get_mask_polygon(image_loaded, image_to_draw_on, indices):
        polygon_positions = FaceFilter.get_mask_polygon_positions(image_loaded, image_to_draw_on, indices)
        mask_image = np.zeros(image_to_draw_on.shape[:2], np.uint8)
        cv2.fillPoly(mask_image, [np.array([polygon_positions], np.int32)], (255, 0, 0))

        return mask_image

    @staticmethod
    def run_face_gausian_filter(image_loaded, image_to_draw_on, size, indices):
        image = image_to_draw_on.copy()
        blur = cv2.blur(image, (int(size), int(size)))
        mask = FaceFilter.get_mask_polygon(image_loaded, image_to_draw_on, indices)

        for y, row in enumerate(mask):
            for x, pixel in enumerate(row):
                if pixel:
                    image_to_draw_on[y, x] = blur[y, x]

        return image_to_draw_on

    @staticmethod
    def run_face_filter_face_features_extraction_interpolation(image_loaded, image_to_draw_on, size, indices):
        mask = FaceFilter.get_mask_polygon(image_loaded, image_to_draw_on, indices)
        image_to_draw_on = cv2.inpaint(image_to_draw_on, mask, int(size), cv2.INPAINT_TELEA)
        return image_to_draw_on

    @staticmethod
    def run_face_cut_features(image_loaded, image_to_draw_on, indices):
        polygon_postions = FaceFilter.get_mask_polygon_positions(image_loaded, image_to_draw_on, indices)
        cv2.fillPoly(image_to_draw_on, [np.array([polygon_postions], np.int32)], (0, 0, 0))
        return image_to_draw_on


    @staticmethod
    def run_face_cut_features2(image_loaded, image_to_draw_on):
        face_landmark_list = face_recognition.face_landmarks(image_to_draw_on)
        print(face_landmark_list)
        for face_landmarks in face_landmark_list:
            top_lip = []
            bottom_lip = []
            right_eye = []
            left_eye = []
            nose_tip = []

            for i in range(12):
                points = list(face_landmarks["top_lip"][i])
                top_lip.append(points)
                points = list(face_landmarks['bottom_lip'][i])
                bottom_lip.append(points)

            for i in range(6):
                points = list(face_landmarks['right_eye'][i])
                right_eye.append(points)
                points = list(face_landmarks['left_eye'][i])
                left_eye.append(points)

            for i in range(5):
                points = list(face_landmarks['nose_tip'][i])
                nose_tip.append(points)

            cv2.fillPoly(image_to_draw_on, [np.array([top_lip], np.int32)], (0, 0, 0))
            cv2.fillPoly(image_to_draw_on, [np.array([bottom_lip], np.int32)], (0, 0, 0))
            cv2.fillPoly(image_to_draw_on, [np.array([right_eye], np.int32)], (0, 0, 0))
            cv2.fillPoly(image_to_draw_on, [np.array([left_eye], np.int32)], (0, 0, 0))
            cv2.fillPoly(image_to_draw_on, [np.array([nose_tip], np.int32)], (0, 0, 0))

        return image_to_draw_on
