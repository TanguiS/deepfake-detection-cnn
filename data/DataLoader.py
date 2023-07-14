from pathlib import Path
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from tensorflow.keras.applications import resnet_v2, xception, efficientnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from log_io.logger import Logger


class DataLoader:
    def __init__(
            self,
            videos_root_dir: Path,
            dataframe_pickle: Path,
            shape: int,
            seed: int,
            arch: str,
            batch_size: int = 32,
            distribution: str = "80-10-10"
    ) -> None:
        super().__init__()
        self.__shape = shape
        self.__log = Logger()

        self.__train_generator, self.__val_generator, self.__test_generator = self.__load_dataset(
            dataframe_pickle,
            videos_root_dir,
            batch_size,
            arch,
            seed,
            distribution
        )

    def __load_dataset(
            self,
            dataframe_pickle: Path,
            videos_root_dir: Path,
            batch_size: int,
            arch: str,
            seed: int,
            distribution: str
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:

        df_train, df_val, df_test = self.__initialize_dataframes(dataframe_pickle, distribution, seed)

        preprocess_func = get_preprocess_function(arch)

        if preprocess_func is None:
            raise ValueError(f"Given arch is not valid : {arch}")

        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_func
        )

        kwargs = {
            'dataframe': df_train,
            'class_mode': 'categorical',
            'x_col': 'index_col',
            'y_col': 'label',
            'batch_size': batch_size,
            'target_size': (self.__shape, self.__shape),
            'directory': videos_root_dir,
            'color_mode': 'rgb',
        }

        train = datagen.flow_from_dataframe(
            **kwargs
        )

        kwargs['dataframe'] = df_val
        val = datagen.flow_from_dataframe(
            **kwargs
        )

        kwargs['dataframe'] = df_test
        test = datagen.flow_from_dataframe(
            **kwargs
        )

        return train, val, test

    def __initialize_dataframes(
            self, dataframe_pickle: Path, distribution: str, seed: int
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        self.__log.info('Initializing and Loading dataframes.')
        if not dataframe_pickle.exists() and dataframe_pickle.is_file():
            raise FileNotFoundError(f"Error with dataframe arg : {dataframe_pickle} does not exists, or not a file.")
        self.__log.info(f"Loading datas from : {dataframe_pickle}")
        dataframe: pd.DataFrame = pd.read_pickle(dataframe_pickle)
        dataframe['index_col'] = dataframe.index
        dataframe['index_col'] = dataframe.index.astype(str)
        dataframe['label'] = dataframe['label'].map({True: 'Fake', False: 'Real'})

        dataframe = balance_dataframe(dataframe)

        split = distribution.split('-')
        if int(split[0]) + int(split[1]) + int(split[2]) > 100:
            raise ValueError("Total distribution can not be > 100")
        train_amount = int((int(split[0]) / 100) * len(dataframe))
        val_amount = int((int(split[1]) / 100) * len(dataframe))
        test_amount = int((int(split[2]) / 100) * len(dataframe))
        self.__log.info(
            f"Using distribution {train_amount} frac for training, {val_amount} frac for validating, {test_amount} frac"
            f"for testing with seed : {seed}"
        )

        test_dataframe = dataframe.sample(n=test_amount, random_state=seed)
        dataframe = dataframe.drop(test_dataframe.index)

        evaluation_dataframe = dataframe.sample(n=val_amount, random_state=seed)
        dataframe = dataframe.drop(evaluation_dataframe.index)

        training_dataframe = dataframe.sample(n=train_amount, random_state=seed)

        return training_dataframe, evaluation_dataframe, test_dataframe

    def summary(self):
        for generator, name in (
                (self.__train_generator, "Train"),
                (self.__val_generator, "Evaluation"),
                (self.__test_generator, "Test")
        ):
            class_labels = generator.class_indices
            class_counts = generator.classes
            class_count_dict = dict()
            for label in class_labels:
                index = class_labels[label]
                count = 0
                for item in class_counts:
                    if item == index:
                        count += 1
                class_count_dict[label] = [index, count]

            self.__log.info(f"{name} generator summary : ")
            for label, count in class_count_dict.items():
                self.__log.info(f" -> Found {count} validated image filenames belonging to {label} class")

    @property
    def train_generator(self):
        return self.__train_generator

    @property
    def validation_generator(self):
        return self.__val_generator

    @property
    def test_generator(self):
        return self.__test_generator


def balance_dataframe(dataframe: DataFrame) -> DataFrame:
    Logger().info("Balancing dataframe...")

    class_counts = dataframe['label'].value_counts()
    minority_class = class_counts.idxmax()
    majority_class = class_counts.idxmin()
    max_count_per_class = class_counts.min()

    minority_class_data = dataframe[dataframe['label'] == minority_class]
    majority_class_data = dataframe[dataframe['label'] == majority_class]

    balanced_minority_class_data = minority_class_data.sample(n=max_count_per_class)
    balanced_dataframe = pd.concat([majority_class_data, balanced_minority_class_data])

    return balanced_dataframe


def get_preprocess_function(model_arch: str):
    if model_arch == 'ResNet152V2':
        return resnet_v2.preprocess_input
    elif model_arch == 'Xception':
        return xception.preprocess_input
    elif model_arch == 'EfficientNetB4':
        return efficientnet.preprocess_input
    raise NotImplementedError(f"This architecture is not implemented: {model_arch}")
