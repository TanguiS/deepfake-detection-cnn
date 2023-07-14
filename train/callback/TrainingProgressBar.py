from typing import Union

from keras.callbacks import Callback
from tqdm import tqdm

from log_io.logger import Logger


class TrainingProgressBar(Callback):
    def __init__(self):
        super().__init__()
        self.__current_batch: Union[int, any] = None
        self.__current_epoch: Union[int, any] = None
        self.__progress_bar: Union[tqdm, any] = None
        self.__total_batches: Union[int, any] = None
        self.__epochs: Union[int, any] = None
        self.__log = Logger()

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.__epochs = self.params['epochs']
        self.__total_batches = self.params['steps']

    def on_train_end(self, logs=None):
        super().on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        self.__current_epoch = epoch
        self.__current_batch = 0
        self.__progress_bar = tqdm(
            total=self.__total_batches,
            unit=" Batch ",
            ncols=100,
            desc=f"Training: epoch[{epoch + 1}/{self.__epochs}]"
        )

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.__metrics_progress(logs)
        self.__progress_bar.close()
        self.__log.info("Epoch completed.")

    def on_batch_begin(self, batch, logs=None):
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super().on_batch_end(batch, logs)
        self.__current_batch += 1
        self.__metrics_progress(logs)
        self.__progress_bar.update()

    def __metrics_progress(self, logs):
        metrics_str = "["
        for metric, value in logs.items():
            metrics_str += f'{metric}: {value:.8f}, '
        metrics_str = metrics_str[0:-2] + "]"
        self.__log.info_file(
            f"Progress: [{self.__current_epoch + 1}/{self.__epochs}]: {self.__current_batch}/{self.__total_batches}"
        )
        self.__log.info(metrics_str)
