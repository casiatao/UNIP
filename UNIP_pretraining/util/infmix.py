import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Compose
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image


def mysort(dataset_path, in1k=False):
    types = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']
    dataset_img_path = []
    for type in types:
        dataset_img_path.extend(glob.glob(os.path.join(dataset_path, type)))
    if not in1k:
        try:
            dataset_img_path = sorted(dataset_img_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        except Exception as e:
            print(f"The name of images in {dataset_path} is not numerical!")
    return dataset_img_path


def pil_loader(path: str, gray=False):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if gray:
            # print('use gray')
            return img.convert('RGB').convert('L').convert('RGB')
        return img.convert('RGB')



class InfMix(Dataset):

    def __init__(self, infpre_path, in1k_path, coco_path, transforms=None, spec_dataset=None, use_in1k=False, per_cls_num=200, 
                 use_coco=False, data_ratio=1.0, rgb_gray=False):
        self.transforms = transforms
        self.use_in1k = use_in1k
        self.rgb_gray = rgb_gray
        self.use_coco = use_coco
        self.spec_dataset = spec_dataset
        self.per_cls_num = per_cls_num
        self.img_list = self._make_dataset(infpre_path, in1k_path, coco_path)
        if data_ratio != 1.0:
            total_data_num = len(self.img_list)
            data_num = int(total_data_num * data_ratio)
            data_sample_interval = int(1 / data_ratio)
            self.img_list = self.img_list[::data_sample_interval]
    
        
    def _make_dataset(self, infpre_path, in1k_path, coco_path):
        """
        infpre_path/
            dataset1_name/
                thermal/
                    image1.jpg
                    image2.jpg
                    ...
                rgb/
                    image1.jpg
                    image2.jpg
            dataset2_name/
                thermal/
                    ...
                rgb/
                    ...
        """
        thermal_img_list = []
        if self.spec_dataset is not None:
            datasets = self.spec_dataset
        else:
            datasets = os.listdir(infpre_path)
        
        for dataset in datasets:
            dataset_path = os.path.join(infpre_path, dataset)
            if not os.path.isdir(dataset_path):
                continue
            thermal_path = os.path.join(dataset_path, r'thermal')
            if not os.path.exists(thermal_path):
                continue
            dataset_thermal_img_list = mysort(thermal_path)
            thermal_img_list.extend(dataset_thermal_img_list)
        
        if self.use_in1k:
            for sub_dataset in os.listdir(in1k_path):
                dataset_thermal_img_list = mysort(os.path.join(in1k_path, sub_dataset), in1k=True)
                if self.per_cls_num != 0:
                    dataset_thermal_img_list = dataset_thermal_img_list[:self.per_cls_num]
                thermal_img_list.extend(dataset_thermal_img_list)
                
        if self.use_coco:
            sub_dataset_list = ['train2017']
            for sub_dataset in sub_dataset_list:
                dataset_thermal_img_list = mysort(os.path.join(coco_path, sub_dataset))
                thermal_img_list.extend(dataset_thermal_img_list)  
                
        return thermal_img_list
    
    
    def __len__(self):
        return len(self.img_list)
    
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        
        img = pil_loader(img_path, gray=self.rgb_gray)
                
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
        
    

if __name__ == '__main__':
    transform_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.3801, 0.3801, 0.3801], std=[0.1871, 0.1871, 0.1871]),
        ])
    infpre_path = r'path/to/infpre'
    in1k_path = r'path/to/imagenet'
    coco_path = r'path/to/coco'
    dataset_train = InfMix(infpre_path=infpre_path, in1k_path=in1k_path, coco_path=coco_path, transforms=transform_train, use_in1k=True, use_coco=True, per_cls_num=200, rgb_gray=True)
    print(len(dataset_train))
