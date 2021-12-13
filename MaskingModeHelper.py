from EnumModeSelector import MaskingOption
from FaceFilter import FaceFilter


class MaskingModeHelper:
    """
    The MaskingModeHelper class checks what masking mode has been selected
    and then uses the appropriate function from the FaceFilter class.
    """
    @staticmethod
    def run_masking_mode(
        masking_mode, image_after_masking, masking_size, interpolation_mode
    ):
        if masking_mode == MaskingOption.default:
            pass
        elif masking_mode == MaskingOption.gaussian_filter:
            image_after_masking = FaceFilter.run_face_gausian_filter(
                image_after_masking,
                masking_size,
                FaceFilter.face_without_forehead_chin_indices,
            )

        elif masking_mode == MaskingOption.extract_face_features:
            image_after_masking = FaceFilter.run_face_cut_features(
                image_after_masking, FaceFilter.left_eye_indices
            )
            image_after_masking = FaceFilter.run_face_cut_features(
                image_after_masking, FaceFilter.right_eye_indices
            )
        elif masking_mode == MaskingOption.accurate_extract_face_features:
            image_after_masking = FaceFilter.run_face_cut_features2(image_after_masking)

        elif masking_mode == MaskingOption.extract_face_features_interpolation:
            image_after_masking = FaceFilter.run_interpolation_mode(
                image_after_masking, masking_size, interpolation_mode
            )
        return image_after_masking
