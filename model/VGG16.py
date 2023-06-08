from typing import Tuple

import plaidml.keras

plaidml.keras.install_backend()

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D


class VGG16:
    def __init__(self, shape: Tuple[int, int, int], show_summary: bool = True) -> None:
        super().__init__()

        self.__VGG16 = self.__setup(shape, show_summary)

    def __setup(self, shape: Tuple[int, int, int], show_summary: bool):
        VGG_16 = Sequential()

        VGG_16.add(Conv2D(input_shape=shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        VGG_16.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        VGG_16.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        VGG_16.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

        VGG_16.add(Flatten())

        VGG_16.add(Dense(4096, activation="relu"))
        VGG_16.add(Dense(4096, activation="relu"))
        VGG_16.add(Dense(1, activation="sigmoid"))

        if show_summary:
            VGG_16.summary()
        return VGG_16

    def compile(self):
        from keras import optimizers

        self.__VGG16.compile(loss='binary_crossentropy',
                             optimizer=optimizers.Adam(lr=0.00005),
                             metrics=['acc'])

    @property
    def model(self):
        return self.__VGG16
