class FaceDetectionMode:
    """
    Face detection mode-selection.
    """
    cnn = "CNN"
    hog = "HOG"


class UploadMode:
    """
    Application upload mode-selection.
    """
    upload_default = "Mode Selection"
    upload_image = "Load Image"
    upload_video = "Load Video"
    upload_live_camera = "Live Camera"


class MaskingOption:
    """
    Masking mode-selection.
    """
    default = "No mask"
    gaussian_filter = "Gaussian filter"
    extract_face_features = "Cut features"
    accurate_extract_face_features = "Accurate cut features"
    extract_face_features_interpolation = "Interpolation features"


class MeshMode:
    """
    Mesh mode-selection.
    """
    mesh_points = "mesh with points"
    mesh_contours = "mesh with contours"
    mesh_triangles = "mesh with triangles"
