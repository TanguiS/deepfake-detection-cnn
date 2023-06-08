import plaidml.keras

plaidml.keras.install_backend()

from pathlib import Path
from typing import Tuple, Any

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DataFrameIterator


class DataLoader:
    def __init__(self, videos_root_dir: Path, dataframe_pickle: Path, shape: int, batch_size: int = 32) -> None:
        super().__init__()
        self.__shape = shape

        self.__train_generator, self.__val_generator = self.__load_dataset(dataframe_pickle, videos_root_dir, batch_size)

    def __load_dataset(
            self,
            dataframe_pickle: Path,
            videos_root_dir: Path,
            batch_size: int
    ) -> Tuple[DataFrameIterator, DataFrameIterator]:
        if not dataframe_pickle.exists() and dataframe_pickle.is_file():
            raise FileNotFoundError(f"Error with dataframe arg : {dataframe_pickle} does not exists, or not a file.")
        dataframe: pd.DataFrame = pd.read_pickle(dataframe_pickle)
        dataframe['index_col'] = dataframe.index
        dataframe['index_col'] = dataframe.index.astype(str)
        dataframe['label'] = dataframe['label'].map({True: 'Fake', False: 'Real'})

        train_datagen = ImageDataGenerator(
            validation_split=0.2,
            rescale=1. / 255
        )

        kwargs = {
            'dataframe': dataframe,
            'class_mode': 'binary',
            'x_col': 'index_col',
            'y_col': 'label',
            'batch_size': batch_size,
            'target_size': (self.__shape, self.__shape),
            'directory': videos_root_dir,
            'color_mode': 'rgb',
            'subset': 'training'
        }

        train = train_datagen.flow_from_dataframe(
            **kwargs
        )

        kwargs['subset'] = 'validation'
        val = train_datagen.flow_from_dataframe(
            **kwargs
        )

        return train, val

    @property
    def train_generator(self):
        return self.__train_generator

    @property
    def val_generator(self):
        return self.__val_generator
