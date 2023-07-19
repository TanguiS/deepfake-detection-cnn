import base64
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from numpy import ndarray

from bench.extract.face import FaceDetectorResult
from bench.extract.face.FaceDetectorResult import face_detection_single_frame
from bench.extract.face.face_align import norm_crop, rezize_from_max_length
from bench.extract.face.yunet import YuNet


def pre_processing_image(
        detection_result: FaceDetectorResult,
        source_image: ndarray,
        ratio_value: float = 1.0,
        max_shape: int = 112
) -> ndarray:
    landmarks_reshaped = np.array(detection_result.landmarks / ratio_value).reshape(5, 2)
    cropped_image = norm_crop(source_image, landmarks_reshaped, image_size=max_shape, mode='arcface')
    return cropped_image


def extract_face(image_path_or_base64: Union[str, Path], face_detector_model: YuNet, max_shape: int = 112) -> ndarray:
    image_cv2_probe = read_image(image_path_or_base64)
    image_cv2_probe_resized, ratio_value_probe = rezize_from_max_length(image_cv2_probe, max_shape)
    face_detection: FaceDetectorResult = face_detection_single_frame(image_cv2_probe_resized, face_detector_model)
    return pre_processing_image(face_detection, image_cv2_probe, ratio_value_probe, max_shape)


def read_image(image_path_or_base64: Union[str, Path]) -> ndarray:
    if isinstance(image_path_or_base64, Path):
        return cv2.imread(str(image_path_or_base64))
    image_bytes = base64.b64decode(image_path_or_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
