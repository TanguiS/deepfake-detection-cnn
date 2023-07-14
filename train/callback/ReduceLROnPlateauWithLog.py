from keras.callbacks import ReduceLROnPlateau

from log_io.logger import Logger
from train.callback.util import capture_message_from_monitored_function


class ReduceLROnPlateauWithLog(ReduceLROnPlateau):

    def __init__(self, monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=1e-4, cooldown=0,
                 min_lr=0, **kwargs):
        super().__init__(monitor, factor, patience, verbose, mode, min_delta, cooldown, min_lr, **kwargs)
        self.__log = Logger()

    def on_epoch_end(self, epoch, logs=None):
        capture_message_from_monitored_function(
            self.__log.info,
            super().on_epoch_end,
            {'epoch': epoch, 'logs': logs}
        )
