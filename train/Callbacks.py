import csv
import pickle
import sys
from io import StringIO
from pathlib import Path
from typing import Union

from keras import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

from log_io.logger import Logger


class ModelStateCheckpoint(Callback):

    def __init__(self, filepath: Path):
        super().__init__()
        self.__log = Logger()
        self.__log.info(f"Saving model to : {filepath}")
        self.__filepath = filepath
        self.model: Model = self.model

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.__log.info(f"\n Epoch: {epoch + 1} saving status")
        self.__log.info(f"Saving model: {self.__filepath.name}...")
        self.model.save(str(self.__filepath), include_optimizer=False)
        self.__log.info("Saved")

        pkl = self.__filepath.with_suffix('.pkl')

        self.__log.info(f"Saving optimizer: {pkl.name}...")
        with open(pkl, 'wb') as f:
            pickle.dump(
                {
                    'optimizer': self.model.optimizer.get_config(),
                    'epoch': epoch + 1
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.__log.info("Saved")
        a = 0


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
            metrics_str += f'{metric}: {value:.4f}, '
        metrics_str = metrics_str[0:-2] + "]"
        self.__log.info_file(
            f"Progress: [{self.__current_epoch + 1}/{self.__epochs}]: {self.__current_batch}/{self.__total_batches}"
        )
        self.__log.info(metrics_str)


class CustomEarlyStopping(EarlyStopping):

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                 restore_best_weights=False):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.__log = Logger()

    def on_epoch_end(self, epoch, logs=None):
        stdout_original = sys.stdout
        sys.stdout = StringIO()
        super().on_epoch_end(epoch, logs)
        captured_message = sys.stdout.getvalue()
        self.__log.info(captured_message)
        sys.stdout = stdout_original

    def on_train_end(self, logs=None):
        stdout_original = sys.stdout
        sys.stdout = StringIO()
        super().on_train_end(logs)
        captured_message = sys.stdout.getvalue()
        self.__log.info(captured_message)
        sys.stdout = stdout_original


class CustomReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=1e-4, cooldown=0,
                 min_lr=0, **kwargs):
        super().__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, **kwargs)
        self.__log = Logger()

    def on_epoch_end(self, epoch, logs=None):
        stdout_original = sys.stdout
        sys.stdout = StringIO()
        super().on_epoch_end(epoch, logs)
        captured_message = sys.stdout.getvalue()
        self.__log.info(captured_message)
        sys.stdout = stdout_original


class CustomCSVLogger(Callback):
    def __init__(self, filename: Path, separator: str = ',', append: bool = False, monitor_val: bool = False):
        super().__init__()
        self.csv_writer = None
        self.__total_batches = None
        self.epoch = None
        self.mode = None
        self.csv_file = None
        self.sep = separator
        self.filename = filename
        self.append = append
        self.monitor_val = monitor_val
        self.data_monitor = ['loss', 'acc'] if not monitor_val else ['val_loss', 'val_acc', 'lr']
        self.init_header = ['epoch', 'batch', 'scaled_batch'] if not monitor_val else ['epoch']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        row = [self.epoch]
        if self.monitor_val:
            self.__save_metrics(logs, row)

    def on_train_begin(self, logs=None):
        self.mode = 'a' if self.append else 'w'
        if not self.append and self.filename.exists():
            self.filename.unlink()
        if not self.filename.exists():
            self.filename.touch()
        self.__total_batches = self.params['steps']
        self.csv_file = open(self.filename, newline='', encoding='utf-8', mode=self.mode)
        self.csv_writer = csv.writer(self.csv_file, delimiter=self.sep)
        if not self.append:
            item_header = self.init_header
            for metric in self.data_monitor:
                item_header.append(metric)
            self.csv_writer.writerow(item_header)
            self.csv_file.flush()

    def on_batch_end(self, batch, logs=None):
        row = [self.epoch, batch, (batch / self.__total_batches) + (self.epoch - 1)]
        if not self.monitor_val:
            self.__save_metrics(logs, row)

    def __save_metrics(self, logs, row):
        for metric in self.data_monitor:
            try:
                row.append(logs[metric])
            except KeyError:
                pass
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
