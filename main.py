from pathlib import Path

from config import config
from data.DataLoader import DataLoader
from model.LaunchModel import launch
from model.VGG16 import VGG16

if __name__ == '__main__':
    subjects = Path('D:\\storage-photos\\subjects')
    df = Path('D:\\storage-photos\\subjects\\dataframe.pkl')
    backup = Path('D:\\backup\\deedfake_pickle')
    batch_size = 32

    shape = (256, 256, 3)

    data = DataLoader(subjects, df, 256, batch_size)
    vgg16 = VGG16(shape)

    launch(data, vgg16, batch_size)
