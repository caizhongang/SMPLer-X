from .images.reid_dataset import ReIDTestDataset
from .images.reid_dataset import ReIDTestDatasetDev

def dataset_entry(config):
    # print('config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
