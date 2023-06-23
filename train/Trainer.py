from keras.callbacks import CallbackList, CSVLogger

from data.DataLoader import DataLoader
from log_io.logger import Logger
from models import ModelBase
from train.Callbacks import ModelStateCheckpoint, TrainingProgressBar, CustomEarlyStopping, CustomReduceLROnPlateau, \
    CustomCSVLogger


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
        model_name = self.__model.keras_model_path.stem
        filepath = self.__model.workspace_path.joinpath(model_name)
        optimizer_callback = ModelStateCheckpoint(filepath=filepath)
        progress_bar = TrainingProgressBar()
        early_stop = CustomEarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=10,
            verbose=1,
            mode='auto'
        )
        learning_rate_reduction = CustomReduceLROnPlateau(
            monitor='val_acc',
            patience=2,
            verbose=1,
            factor=0.1,
            min_lr=0.000000001
        )
        filepath = self.__model.workspace_path.joinpath("train_history.csv")
        append = True
        if not filepath.exists():
            append = False
        if self.__model.is_first_run() and filepath.exists():
            append = False
        train_csv_save = CustomCSVLogger(filepath, separator=';', append=append)
        filepath = self.__model.workspace_path.joinpath("val_history.csv")
        val_csv_save = CustomCSVLogger(filepath, separator=';', append=append, monitor_val=True)

        self.__callbacks = CallbackList(
            [optimizer_callback, progress_bar, early_stop, learning_rate_reduction, train_csv_save, val_csv_save]
        )

    def run(self, max_epochs: int = 10):
        self.__model.compile()

        self.__model.keras_model().fit_generator(
            self.__train_data,
            steps_per_epoch=self.__train_data.samples // self.__batch_size,
            validation_data=self.__val_data,
            validation_steps=self.__val_data.samples // self.__batch_size,
            epochs=max_epochs,
            initial_epoch=self.__model.start_epoch,
            callbacks=[self.__callbacks],
            verbose=0
        )

        test_steps = self.__test_data.samples // self.__batch_size
        results = self.__model.keras_model().evaluate_generator(self.__test_data, steps=test_steps, verbose=1)
        self.__log.info(results)