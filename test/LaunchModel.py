import plaidml.keras

plaidml.keras.install_backend()

import matplotlib.pyplot as plt
import keras.callbacks

from pathlib import Path
from data.DataLoader import DataLoader
from test.VGG16 import VGG16
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class ModelCheckpointSpaceSaver(keras.callbacks.ModelCheckpoint):

    def __init__(self, root_dir: Path, model_name: str, model_purpose: str, monitor='val_loss', verbose=0,
                 save_weights_only=False, mode='auto', period=1, max_saved_model: int = 10):

        filepath = str(root_dir.joinpath(f"{model_name}_{model_purpose}_{{epoch:02d}}.h5"))
        super().__init__(filepath, monitor, verbose, False, save_weights_only, mode, period)
        self.__max_saved_model = max_saved_model
        self.__root_dir = root_dir
        self.__save_pattern = f"{model_name}_{model_purpose}"

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        models = [file for file in self.__root_dir.glob(f"{self.__save_pattern}*")]
        if len(models) <= self.__max_saved_model:
            pass
        models.sort()
        while len(models) > self.__max_saved_model:
            to_unlink = models.pop(0)
            to_unlink.unlink()


def launch(data_loader: DataLoader, vgg_16: VGG16, batch_size: int):
    checkpoint = ModelCheckpointSpaceSaver(
        root_dir=Path('./log'),
        model_name='VGG16',
        model_purpose="deepfake-detection",
        monitor='val_acc',
        verbose=1,
        save_weights_only=False,
        mode='auto',
        period=1,
        max_saved_model=1
    )
    early = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          mode='auto')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=2,
                                                verbose=1,
                                                factor=0.1,
                                                min_lr=0.000000001)

    nb_epochs = 2

    train_generator = data_loader.train_generator
    validation_generator = data_loader.validation_generator
    test_generator = data_loader.test_generator

    vgg_16.compile()
    VGG_16 = vgg_16.model

    history = VGG_16.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=nb_epochs,
        callbacks=[checkpoint, early, learning_rate_reduction]
    )

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('models accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('models loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Metrics testing
    test_steps = test_generator.samples // batch_size
    results = VGG_16.evaluate_generator(test_generator, steps=test_steps, verbose=1)
    print(results)
