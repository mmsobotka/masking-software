class FaceDetectionMode:
    cnn = "CNN"
    hog = "HOG"
    lbph = "LBPH"


class UploadMode:
    upload_default = "Mode Selection"
    upload_image = "Load Image"
    upload_video = "Load Video"
    upload_live_camera = "Live Camera"


class MaskingOption:
    default = "No mask"
    gaussian_filter = "Gaussian filter"
    extract_face_features = "Cut features"
    accurate_extract_face_features = "Accurate cut features"
    # face_transform = "Face transform"
    extract_face_features_interpolation = "Interpolation features"


class MeshMode:
    mesh_points = "mesh with points"
    mesh_contours = "mesh with contours"
    mesh_triangles = "mesh with triangles"
