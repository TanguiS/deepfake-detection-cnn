from tensorflow.keras.callbacks import CallbackList

from data.DataLoader import DataLoader
from log_io.logger import Logger
from models import ModelBase
from train.callback.ModelStateCheckpoint import ModelStateCheckpoint
from train.callback.TrainingProgressBar import TrainingProgressBar
from train.callback.EarlyStoppingWithLog import EarlyStoppingWithLog
from train.callback.ReduceLROnPlateauWithLog import ReduceLROnPlateauWithLog
from train.callback.CSVHistoryLogger import CSVHistoryLogger


class Trainer:
    def __init__(self, model: ModelBase, data_loader: DataLoader, batch_size: int = 32) -> None:
        super().__init__()
        self.__model = model
        self.__train_data = data_loader.train_generator
        self.__val_data = data_loader.validation_generator
        self.__test_data = data_loader.test_generator
        self.__batch_size = batch_size
        self.__log = Logger()
        self.__setup_callbacks()

    def __setup_callbacks(self):
        filepath = self.__model.get_keras_model_path
        optimizer_callback = ModelStateCheckpoint(filepath=filepath)
        progress_bar = TrainingProgressBar()
        early_stop = EarlyStoppingWithLog(
            monitor='val_accuracy',
            min_delta=0,
            patience=10,
            verbose=1,
            mode='auto'
        )
        learning_rate_reduction = ReduceLROnPlateauWithLog(
            monitor='val_accuracy',
            patience=2,
            verbose=1,
            factor=0.1,
            min_lr=0.000000001
        )
        name = self.__model.get_keras_model_path.stem
        filepath = self.__model.get_workspace_path.joinpath(f"{name}_train_history.csv")
        append = True
        if not filepath.exists():
            append = False
        if self.__model.is_first_run:
            append = False
        train_csv_save = CSVHistoryLogger(filepath, separator=';', append=append)
        filepath = self.__model.get_workspace_path.joinpath(f"{name}_val_history.csv")
        val_csv_save = CSVHistoryLogger(filepath, separator=';', append=append, monitor_val=True)

        self.__callbacks = [
            optimizer_callback, progress_bar, early_stop, learning_rate_reduction, train_csv_save, val_csv_save
        ]

    def run(self, max_epochs: int = 10):
        self.__model.compile()

        if self.__model.get_start_epoch >= max_epochs:
            self.__log.err(f"Error: start_epoch[{self.__model.get_start_epoch}] >= max_epoch[{max_epochs}].")
            raise ValueError

        if self.__model.get_start_epoch > 0 and not self.__model.is_first_run:
            self.__log.info(f"Resume training, starting at {self.__model.get_start_epoch} epoch to {max_epochs}.")
        else:
            self.__log.info(f"Starting training from scratch to {max_epochs} epochs.")

        self.__model.get_keras_model.fit(
            self.__train_data,
            steps_per_epoch=self.__train_data.samples // self.__batch_size,
            validation_data=self.__val_data,
            validation_steps=self.__val_data.samples // self.__batch_size,
            epochs=max_epochs,
            initial_epoch=self.__model.get_start_epoch,
            callbacks=self.__callbacks,
            verbose=0
        )

        self.__log.info(f"TRAINING Done: Saving model without Optimizer to {self.__model.get_keras_model_path.name}...")
        self.__model.get_keras_model.save(str(self.__model.get_keras_model_path), include_optimizer=False)
        self.__log.info(" > Saved")

        self.__log.info("Testing model on Test generator.")
        test_steps = self.__test_data.samples // self.__batch_size
        results = self.__model.get_keras_model.evaluate(self.__test_data, steps=test_steps, verbose=1)
        self.__log.info(f"Results of tests : {results}")
