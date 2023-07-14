from pathlib import Path
from typing import Tuple, List, Union

from bench.Evaluation import ModelEvaluator


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
