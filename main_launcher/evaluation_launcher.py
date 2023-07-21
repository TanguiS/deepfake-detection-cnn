import base64
import csv
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Generator, List

import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import models
from bench.Evaluation import ModelEvaluator
from bench.extract.face.FaceDetectorResult import load_face_detection_model
from bench.util import is_prediction_deepfake
from data.DataLoader import balance_dataframe
from env.config import cpu_config
from log_io.logger import Logger


def decode_evaluation_args(args: Dict[str, any]):
    from main_launcher.util import decode_shape
    input_shape = decode_shape(args['shape'], args['grayscale'])
    kwargs = {
        'input_shape': input_shape,
        'root_face_folder': args['root_face_folder'],
        'df_faces_path': args['df_faces_path'],
        'output_folder': args['output_folder'],
        'arch': args['arch'],
        'model_name': args['model_name'],
        'seed': args['seed'],
        'distribution': args['distribution'],
        'yunet_model_path': args['yunet_model_path']
    }
    return kwargs


def frame_to_base64(frame_path: Path) -> str:
    image = cv2.imread(str(frame_path))
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return base64_image


def test_sample_base64_iterator(
        df_path: Path, root: Path, distribution: str, seed: int
) -> Generator[Tuple[str, bool], None, None]:
    df = pd.read_pickle(df_path)
    dataframe = balance_dataframe(df)
    split = distribution.split('-')
    if int(split[0]) + int(split[1]) + int(split[2]) > 100:
        raise ValueError("Total distribution can not be > 100")
    test_amount = int((int(split[2]) / 100) * len(dataframe))
    test_frame = dataframe.sample(n=test_amount, random_state=seed)

    for frame_line in test_frame.iterrows():
        yield frame_to_base64(root.joinpath(frame_line[0])), frame_line[1]['label']


def evaluate_model(
        root_face_folder: Path, df_faces_path: Path, arch: str, input_shape: Tuple[int, int, int],
        output_folder: Path, distribution: str, seed: int, yunet_model_path: Path, model_name: Optional[str]
) -> Tuple[List[int], List[int], float, List[float], List[float]]:
    yunet = load_face_detection_model(yunet_model_path, input_size=(input_shape[0], input_shape[1]))

    model = models.import_model(arch)(
        models_dir=output_folder,
        model_arch=arch,
        input_shape=input_shape,
        model_name=model_name
    )
    model.compile()

    evaluator = ModelEvaluator(model, yunet)
    count = 0
    sum_time = 0
    y_true = []
    y_pred = []
    yhat_0 = []
    yhat_1 = []

    print("Evaluating...")
    for frame, label in tqdm(test_sample_base64_iterator(df_faces_path, root_face_folder, distribution, seed)):
        start = time.time()
        yhat = evaluator.true_evaluate(frame)
        end = time.time()

        count += 1
        sum_time += (end - start)

        y_true.append(int(label))
        y_pred.append(int(is_prediction_deepfake(yhat)))

        yhat_0.append(yhat[0][0])
        yhat_1.append(yhat[0][1])

    return y_true, y_pred, sum_time / count, yhat_0, yhat_1


def plot_confusion_matrix(y_true: list, y_pred: list, arch: str, shape: Tuple[int, int, int]) -> None:
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix: {arch} - {shape}')
    plt.show()


def save_yhat(yhat_0: List[float], yhat_1: List[float], sum_time: float, csv_path: Path) -> None:
    data = list(zip(yhat_0, yhat_1))

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        csv_writer.writerow(['Yhat Fake', 'Yhat Real'])
        csv_writer.writerows(data)
        csv_writer.writerow(['', '', 'Sum Time', sum_time])


def launch_evaluation(
        root_face_folder: Path,
        df_faces_path: Path,
        arch: str,
        input_shape: Tuple[int, int, int],
        output_folder: Path,
        distribution: str,
        seed: int,
        yunet_model_path: Path,
        model_name: Optional[str]
) -> None:
    cpu_config()
    workspace = output_folder.joinpath(arch)
    workspace.mkdir(exist_ok=True)
    logger = Logger()

    y_true, y_pred, avg_prediction_time, yhat_0, yhat_1 = evaluate_model(
        root_face_folder, df_faces_path, arch, input_shape, output_folder,
        distribution, seed, yunet_model_path, model_name
    )

    plot_confusion_matrix(y_true, y_pred, arch, input_shape)
    print(f"Average time to make a prediction: {avg_prediction_time}")

    save_yhat(yhat_0, yhat_1, avg_prediction_time, workspace.joinpath('yhat.csv'))
