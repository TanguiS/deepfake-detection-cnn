from pathlib import Path
from typing import Tuple, Union, List

import numpy as np

from bench.extract.face import face_extract
from bench.extract.face.yunet import YuNet
from data.DataLoader import get_preprocess_function
from models import ModelBase


class ModelEvaluator:
    def __init__(self, compiled_model: ModelBase, face_detector_model: YuNet) -> None:
        super().__init__()
        self.__model = compiled_model
        self.__face_detector = face_detector_model
        self.__shape = self.__model.get_input_shape

    def evaluate(self, frame_path: Path, face_detector_model: YuNet):
        face = face_extract.extract_face(frame_path, face_detector_model, self.__shape[0])
        x = np.expand_dims(face / 255, axis=0)
        images = np.vstack([x])
        yhat = self.__model.get_keras_model.predict(
            images
        )
        return yhat

    def true_evaluate(self, frame_path_or_base64: Union[str, Path]):
        extracted_frame = face_extract.extract_face(
            frame_path_or_base64, self.__face_detector, self.__model.get_input_shape[0]
        )
        preprocess_func = get_preprocess_function(self.__model.get_arch)
        preprocessed_frame = preprocess_func(extracted_frame)
        yhat = self.__model.get_keras_model.predict(preprocessed_frame)
        return yhat


def decode_prediction(prediction) -> Tuple[str, float]:
    label = 'Real'
    if prediction[0][0] <= 0.5:
        label = 'Fake'
    confidence = (abs(prediction[0][0] - 0.5) * 100) / 0.5
    return label, confidence


def is_prediction_deepfake(prediction) -> bool:
    return prediction[0][0] <= 0.5


def predict_list(evaluator: ModelEvaluator, list_frame_path_or_base64: List[Union[str, Path]]):
    predictions = []
    for frame in list_frame_path_or_base64:
        predictions.append(evaluator.true_evaluate(frame))
    return predictions


def decode_predictions_list(predictions: List) -> List[Tuple[str, float]]:
    decoded_pred = []
    for prediction in predictions:
        label = 'Real'
        if prediction[0][0] <= 0.5:
            label = 'Fake'
        confidence = (abs(prediction[0][0] - 0.5) * 100) / 0.5
        decoded_pred.append((label, confidence))
    return decoded_pred


def is_prediction_list_has_deepfake(predictions: List) -> bool:
    for prediction in predictions:
        if is_prediction_deepfake(prediction):
            return True
