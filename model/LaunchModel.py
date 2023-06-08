import plaidml.keras

plaidml.keras.install_backend()

from data.DataLoader import DataLoader
from model.VGG16 import VGG16
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def launch(data_loader: DataLoader, vgg_16: VGG16, batch_size: int):
    checkpoint = ModelCheckpoint("VGG16_dog_cat.h5",
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
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

    nb_epochs = 10

    train_generator = data_loader.train_generator
    validation_generator = data_loader.val_generator

    vgg_16.compile()
    VGG_16 = vgg_16.model

    history = VGG_16.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=nb_epochs,
        callbacks=[checkpoint, early, learning_rate_reduction])


    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
