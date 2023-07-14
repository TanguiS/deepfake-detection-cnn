from keras.callbacks import EarlyStopping

from log_io.logger import Logger
from train.callback.util import capture_message_from_monitored_function


class EarlyStoppingWithLog(EarlyStopping):

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                 restore_best_weights=False):
        super().__init__(monitor, min_delta, patience, verbose, mode, baseline, restore_best_weights)
        self.__log = Logger()

    def on_epoch_end(self, epoch, logs=None):
        capture_message_from_monitored_function(self.__log.info, super().on_epoch_end, {'epoch': epoch, 'logs': logs})

    def on_train_end(self, logs=None):
        capture_message_from_monitored_function(self.__log.info, super().on_train_end, {'logs': logs})
