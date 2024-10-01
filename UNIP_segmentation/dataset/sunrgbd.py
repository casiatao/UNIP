from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

import os.path as osp



@DATASETS.register_module()
class SUNRGBD(CustomDataset):
    METAINFO = dict(
        classes=(
            "_background_",
            "wall",
            "floor",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "bookshelf",
            "picture",
            "counter",
            "blinds",
            "desk",
            "shelves",
            "curtain",
            "dresser",
            "pillow",
            "mirror",
            "floor_mat",
            "clothes",
            "ceiling",
            "books",
            "fridge",
            "tv",
            "paper",
            "towel",
            "shower_curtain",
            "box",
            "whiteboard",
            "person",
            "night_stand",
            "toilet",
            "sink",
            "lamp",
            "bathtub",
            "bag"),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255]])
    
    def __init__(self, split, **kwargs):
        super(SUNRGBD, self).__init__(img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None