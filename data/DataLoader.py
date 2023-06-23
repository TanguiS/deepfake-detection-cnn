import plaidml.keras


plaidml.keras.install_backend()

from log_io.logger import Logger
from pathlib import Path
from typing import Tuple

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DataFrameIterator


class DataLoader:
    def __init__(self, videos_root_dir: Path, dataframe_pickle: Path, shape: int, batch_size: int = 32) -> None:
        super().__init__()
        self.__shape = shape
        self.__log = Logger()

        self.__train_generator, self.__val_generator, self.__test_generator = self.__load_dataset(
            dataframe_pickle,
            videos_root_dir,
            batch_size
        )

    def __load_dataset(
            self,
            dataframe_pickle: Path,
            videos_root_dir: Path,
            batch_size: int
    ) -> Tuple[DataFrameIterator, DataFrameIterator, DataFrameIterator]:
        if not dataframe_pickle.exists() and dataframe_pickle.is_file():
            raise FileNotFoundError(f"Error with dataframe arg : {dataframe_pickle} does not exists, or not a file.")
        dataframe: pd.DataFrame = pd.read_pickle(dataframe_pickle)
        dataframe['index_col'] = dataframe.index
        dataframe['index_col'] = dataframe.index.astype(str)
        dataframe['label'] = dataframe['label'].map({True: 'Fake', False: 'Real'})

        test_dataframe = dataframe.sample(frac=0.1)
        dataframe = dataframe.drop(test_dataframe.index)

        evaluation_dataframe = dataframe.sample(frac=0.2)
        dataframe = dataframe.drop(evaluation_dataframe.index)

        datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        kwargs = {
            'dataframe': dataframe,
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

        kwargs['dataframe'] = evaluation_dataframe
        val = datagen.flow_from_dataframe(
            **kwargs
        )

        kwargs['dataframe'] = test_dataframe
        test = datagen.flow_from_dataframe(
            **kwargs
        )

        return train, val, test

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
