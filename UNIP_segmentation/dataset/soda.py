from mmseg.datasets.builder import DATASETS
from mmseg.datasets.voc import PascalVOCDataset


@DATASETS.register_module()
class SODA(PascalVOCDataset):
    CLASSES = (
    "_background_",
    "person",
    "building",
    "tree",
    "road",
    "pole",
    "grass",
    "door",
    "table",
    "chair",
    "car",
    "bicycle",
    "lamp",
    "monitor",
    "trafficCone",
    "trash can",
    "animal",
    "fence",
    "sky",
    "river",
    "sidewalk")
    
    def __init__(self, split, **kwargs):
        super().__init__(split, **kwargs)