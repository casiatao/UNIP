from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp


@DATASETS.register_module()
class MFNet(CustomDataset):
    
    METAINFO = dict(
        classes=("_background_", "car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0]])
    
    def __init__(self, split, **kwargs):
        super(MFNet, self).__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None