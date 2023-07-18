from pathlib import Path
from typing import Tuple, List, Union

from bench.Evaluation import ModelEvaluator


def is_prediction_deepfake(prediction) -> bool:
    return prediction[0][0] > prediction[0][1]


def major_prediction_to_percentage(prediction) -> float:
    if is_prediction_deepfake(prediction):
        return float(int(prediction[0][0] * 10000)) / 10
    return float(int(prediction[0][1] * 10000)) / 10


def decode_prediction(prediction) -> Tuple[str, float]:
    return 'Fake' if is_prediction_deepfake(prediction) else 'Real', major_prediction_to_percentage(prediction)


def predict_list(evaluator: ModelEvaluator, list_frame_path_or_base64: List[Union[str, Path]]):
    predictions = []
    for frame in list_frame_path_or_base64:
        predictions.append(evaluator.true_evaluate(frame))
    return predictions


def decode_predictions_list(predictions: List) -> List[Tuple[str, float]]:
    decoded_pred = []
    for prediction in predictions:
        decoded_pred.append(decode_prediction(prediction))
    return decoded_pred


def is_prediction_list_has_deepfake(predictions: List) -> bool:
    for prediction in predictions:
        if is_prediction_deepfake(prediction):
            return True
