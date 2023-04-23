import random
import numpy as np
from mmdet.datasets import DATASETS
from mmdet.datasets import build_dataset

from mmcv.utils import get_logger
from .custom_3d import Custom3DDataset

@DATASETS.register_module()
class LabeledDataset(Custom3DDataset):
    """Labeled dataset for semi-supervised 3D object detection task.

    Args:
        Custom3DDataset (_type_): _description_
    """
    
    def __init__(self,
                 seed,
                 src,
                 ratio):
        self.src_cfg = src
        self.src = build_dataset(src)
        self.CLASSES = self.src.CLASSES
        self.flag = self.src.flag
        
        self.ratio = ratio
        self.total_len = len(self.src.data_infos)
        
        # Generate Mapping
        random.seed(seed)
        self.mapping = list(range(self.total_len))
        random.shuffle(self.mapping)
        
        self.len = int(self.total_len * self.ratio)
        self.len = max(self.len, 1)
        self.mapping = self.mapping[:self.len]
        
        logger = get_logger("SemiDataset")
        logger.info(f"Total {self.len} samples are labeled.")
    
    
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, idx):
        return self.src[self.mapping[idx]]


@DATASETS.register_module()
class UnlabeledDataset(Custom3DDataset):
    """Unlabeled dataset for semi-supervised 3D object detection task.

    Args:
        Custom3DDataset (_type_): _description_
    """
    
    def __init__(self,
                 seed,
                 src,
                 ratio,
                 drop_gt=True):
        self.src_cfg = src
        self.src = build_dataset(src)
        self.CLASSES = self.src.CLASSES
        self.ratio = ratio
        
        self.flag = self.src.flag
        
        self.total_len = len(self.src.data_infos)
        
        # Generate Mapping
        random.seed(seed)
        self.mapping = list(range(self.total_len))
        random.shuffle(self.mapping)
        
        self.len = int(self.total_len * self.ratio)
        self.len = min(self.len, self.total_len-1)
        self.mapping = self.mapping[self.len:]
        
        self.drop_gt = drop_gt
        
        logger = get_logger("SemiDataset")
        logger.info(f"Total {self.total_len - self.len} samples are unlabeled.")
    
    
    def __len__(self):
        return self.total_len - self.len
    
    def __getitem__(self, idx):
        ret_candidate = self.src[self.mapping[idx]]
        
        if self.drop_gt:
            # We filter the ret_candidate to ignore gt*
            return {
                k: v for k, v in ret_candidate.items()
                if "gt" not in k
            }
        else:
            return ret_candidate

@DATASETS.register_module()
class SemiDataset(Custom3DDataset):
    def __init__(self, labeled, unlabeled):
        self.labeled_cfg = labeled
        self.unlabeled_cfg = unlabeled
        
        self.labeled = build_dataset(labeled)
        self.unlabeled = build_dataset(unlabeled)
        
        self.CLASSES = self.labeled.CLASSES
        self.flag = np.zeros(len(self), dtype=np.uint8)
        
    def __len__(self):
        return len(self.labeled) * len(self.unlabeled)
    
    def __getitem__(self, idx):
        s1 = self.labeled[idx // len(self.unlabeled)]
        s2 = self.unlabeled[idx % len(self.unlabeled)]
        return {
            **s1,
            "unlabeled_data": s2
        }
