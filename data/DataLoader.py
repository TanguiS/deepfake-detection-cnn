import plaidml.keras
from pandas import DataFrame

plaidml.keras.install_backend()

from log_io.logger import Logger
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DataFrameIterator


class DataLoader:
    def __init__(
            self,
            videos_root_dir: Path,
            dataframe_pickle: Path,
            shape: int,
            save_dir: Path,
            batch_size: int = 32
    ) -> None:
        super().__init__()
        self.__shape = shape
        self.__log = Logger()
        self.__save_dir = save_dir
        self.__DF_TRAIN_PATH = self.__save_dir.joinpath("train_dataframe.pkl")
        self.__DF_VAL_PATH = self.__save_dir.joinpath("validation_dataframe.pkl")
        self.__DF_TEST_PATH = self.__save_dir.joinpath("test_dataframe.pkl")

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

        dataframes = self.__load_dataframes()
        if dataframes is None:
            self.__log.info('Failed to load dataframes from the saving directory')
            df_train, df_val, df_test = self.__initialize_dataframes(dataframe_pickle)
            self.__save_dataframes(df_train, df_val, df_test)
        else:
            df_train, df_val, df_test = dataframes

        datagen = ImageDataGenerator(
            rescale=1. / 255
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

    def __initialize_dataframes(self, dataframe_pickle):
        self.__log.info('Initializing and Loading dataframes.')
        if not dataframe_pickle.exists() and dataframe_pickle.is_file():
            raise FileNotFoundError(f"Error with dataframe arg : {dataframe_pickle} does not exists, or not a file.")
        self.__log.info(f"Loading datas from : {dataframe_pickle}")
        dataframe: pd.DataFrame = pd.read_pickle(dataframe_pickle)
        dataframe['index_col'] = dataframe.index
        dataframe['index_col'] = dataframe.index.astype(str)
        dataframe['label'] = dataframe['label'].map({True: 'Fake', False: 'Real'})
        test_dataframe = dataframe.sample(frac=0.1)
        dataframe = dataframe.drop(test_dataframe.index)
        evaluation_dataframe = dataframe.sample(frac=0.2)
        dataframe = dataframe.drop(evaluation_dataframe.index)
        return dataframe, evaluation_dataframe, test_dataframe

    def __save_dataframes(self, df_train: DataFrame, df_val: DataFrame, df_test: DataFrame):
        self.__log.info(f"Saving Dataframe for reproducibility to : {self.__save_dir}")
        self.__save_dir.mkdir(parents=True, exist_ok=True)
        if self.__DF_TRAIN_PATH.exists():
            self.__DF_TRAIN_PATH.unlink()
        if self.__DF_VAL_PATH.exists():
            self.__DF_VAL_PATH.unlink()
        if self.__DF_TEST_PATH.exists():
            self.__DF_TEST_PATH.unlink()
        df_train.to_pickle(str(self.__DF_TRAIN_PATH))
        df_val.to_pickle(str(self.__DF_VAL_PATH))
        df_test.to_pickle(str(self.__DF_TEST_PATH))

    def __load_dataframes(self) -> Optional[Tuple[DataFrame, DataFrame, DataFrame]]:
        self.__log.info(f'Trying to load dataframes from {self.__DF_TRAIN_PATH.parents}')
        if not self.__DF_TRAIN_PATH.exists():
            return None
        if not self.__DF_VAL_PATH.exists():
            return None
        if not self.__DF_TEST_PATH.exists():
            return None
        df_train = pd.read_pickle(self.__DF_TRAIN_PATH)
        df_val = pd.read_pickle(self.__DF_VAL_PATH)
        df_test = pd.read_pickle(self.__DF_TEST_PATH)
        return df_train, df_val, df_test

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
