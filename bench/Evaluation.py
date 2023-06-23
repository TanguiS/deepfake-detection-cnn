import plaidml.keras

from bench.extract.face import face_extract
from bench.extract.face.yunet import YuNet

plaidml.keras.install_backend()

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image


class ModelEvaluator:
    def __init__(self, model_path: Path, shape: Tuple[int, int]) -> None:
        super().__init__()
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"Path to model is invalid, either not a file or does not exists : {model_path}")
        self.__model = load_model(str(model_path))
        self.__shape = shape
        # test = self.__model.get

    def evaluate(self, frame_path: Path, face_detector_model: YuNet):
        face = face_extract.extract_face(frame_path, face_detector_model, self.__shape[0])
        x = np.expand_dims(face/255, axis=0)
        images = np.vstack([x])
        yhat = self.__model.predict(
            images
        )
        return yhat


def decode_prediction(prediction) -> Tuple[str, float]:
    label = 'Real'
    if prediction[0][0] <= 0.5:
        label = 'Fake'
    confidence = (abs(prediction[0][0] - 0.5) * 100) / 0.5
    return label, confidence
