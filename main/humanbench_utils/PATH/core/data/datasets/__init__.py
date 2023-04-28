from .images.reid_dataset import ReIDDataset
from .images.pedattr_dataset import AttrDataset
from .images.pos_dataset_dev import COCOPosDatasetDev, MPIIPosDatasetDev
from .images.multi_posedataset import MultiPoseDatasetDev
from .images.parsing_dataset import Human3M6ParsingDataset, LIPParsingDataset, CIHPParsingDataset, ATRParsingDataset, DeepFashionParsingDataset, VIPParsingDataset, ModaNetParsingDataset, PaperDollParsingDataset
from .images.pedattr_dataset import AttrDataset, MultiAttrDataset
from .images.peddet_dataset import PedestrainDetectionDataset
from core.utils import printlog

def dataset_entry(config):
    printlog('config[kwargs]',config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
