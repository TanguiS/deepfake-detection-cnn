import plaidml.keras

plaidml.keras.install_backend()


if __name__ == '__main__':
    from log_io.logger import Logger
    from models import ModelBase
    import pandas as pd
    from train.Trainer import Trainer
    from pathlib import Path
    import models
    from data.DataLoader import DataLoader

    subjects = Path('C:\\WORK\\subjects')
    df = Path('C:\\WORK\\subjects\\dataframe.pkl')
    models_dir = Path('C:\\WORK\\deepfake-detection-cnn\\log')
    df_save = Path('C:\\WORK\\deepfake-detection-cnn\\log\\df')

    shape = (256, 256, 3)
    batch_size = 32

    workspace = models_dir.joinpath('VGG19')
    workspace.mkdir(exist_ok=True)
    logger = Logger(workspace)

    data = DataLoader(subjects, df, shape[0], df_save, batch_size)
    data.summary()

    model: ModelBase = models.import_model('VGG19')(
        models_dir=models_dir,
        model_arch='VGG19',
        input_shape=shape,
        nb_epoch=1,
        batch_size=batch_size,
        model_name=None
    )
    model.show_summary()

    trainer = Trainer(model, data, batch_size)

    trainer.run(100)
