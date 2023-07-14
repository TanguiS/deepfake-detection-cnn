import csv
import time
from pathlib import Path

from keras.callbacks import Callback


class CSVHistoryLogger(Callback):
    def __init__(self, filename: Path, separator: str = ',', append: bool = False, monitor_val: bool = False):
        super().__init__()
        self.time_epoch_begin = 0
        self.csv_writer = None
        self.__total_batches = None
        self.epoch = None
        self.mode = None
        self.csv_file = None
        self.sep = separator
        self.filename = filename
        self.append = append
        self.monitor_val = monitor_val
        self.time_batch_begin = 0
        self.rows = []
        self.data_monitor = ['loss', 'accuracy'] if not monitor_val else ['val_loss', 'val_accuracy', 'lr']
        self.init_header = ['epoch', 'time (s)', 'batch', 'scaled_batch'] if not monitor_val else ['epoch', 'time (s)']

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

    def on_train_end(self, logs=None):
        self.csv_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.time_epoch_begin = time.time()
        self.epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        time_epoch_end = time.time()
        elapsed_time = time_epoch_end - self.time_epoch_begin
        row = [self.epoch, elapsed_time]
        if self.monitor_val:
            self.__add_metrics(logs, row)
            self.__save_metrics(row)
            return
        for row in self.rows:
            self.__save_metrics(row)
        self.rows.clear()

    def on_batch_begin(self, batch, logs=None):
        self.time_batch_begin = time.time()
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        time_batch_end = time.time()
        elapsed_time = time_batch_end - self.time_batch_begin
        row = [self.epoch, elapsed_time, batch, (batch / self.__total_batches) + (self.epoch - 1)]
        if not self.monitor_val:
            self.__add_metrics(logs, row)
            self.rows.append(row)

    def __add_metrics(self, logs, row):
        for metric in self.data_monitor:
            try:
                row.append(logs[metric])
            except KeyError:
                pass

    def __save_metrics(self, row):
        self.csv_writer.writerow(row)
        self.csv_file.flush()
