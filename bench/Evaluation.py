from pathlib import Path
from typing import Tuple

import numpy as np

from bench.extract.face import face_extract
from bench.extract.face.yunet import YuNet
from models import ModelBase


class ModelEvaluator:
    def __init__(self, compiled_model: ModelBase) -> None:
        super().__init__()
        self.__model = compiled_model
        self.__shape = self.__model.input_shape

    def evaluate(self, frame_path: Path, face_detector_model: YuNet):
        face = face_extract.extract_face(frame_path, face_detector_model, self.__shape[0])
        x = np.expand_dims(face/255, axis=0)
        images = np.vstack([x])
        yhat = self.__model.keras_model().predict(
            images
        )
        return yhat


def decode_prediction(prediction) -> Tuple[str, float]:
    label = 'Real'
    if prediction[0][0] <= 0.5:
        label = 'Fake'
    confidence = (abs(prediction[0][0] - 0.5) * 100) / 0.5
    return label, confidence
