import pickle
from pathlib import Path

from keras import Model
from keras.callbacks import Callback

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
