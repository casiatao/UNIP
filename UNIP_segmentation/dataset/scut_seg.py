from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp


@DATASETS.register_module()
class SCUT_seg(CustomDataset):
    METAINFO = dict(
        classes=("background", "road", "person", "rider", "car", "truck", "fence", "tree", "bus", "pole"),
        palette=[[0, 0, 0], [128, 64, 128], [60, 20, 220], [0, 0, 255], [142, 0, 0],
               [70, 0, 0], [153, 153, 190], [35, 142, 107], [100, 60, 0], [153, 153, 153]])
    
    def __init__(self, split, **kwargs):
        super(SCUT_seg, self).__init__(img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None