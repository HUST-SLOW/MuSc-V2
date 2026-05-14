import os
from PIL import Image
from torchvision import transforms
import glob
from utils.tools import *
import numpy as np
import torch
from enum import Enum
import PIL

_CLASSNAMES = [
    'CandyCane',
    'ChocolateCookie',
    'ChocolatePraline',
    'Confetto',
    'GummyBear',
    'HazelnutTruffle',
    'LicoriceSandwich',
    'Lollipop',
    'Marshmallow',
    'PeppermintCandy',
]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

def pc_normalize_mvtec(pc):
    pc = pc.permute(1, 2, 0)
    w, h, c = pc.shape
    pc = pc.reshape(-1, 3)
    # (N, 3)
    nonzeros_indices = torch.nonzero(torch.all(pc != 0, dim=1)).squeeze()
    pc_nonzeros = pc[nonzeros_indices, :]
    centroid = pc_nonzeros.mean(0)
    pc_nonzeros = pc_nonzeros - centroid
    m = torch.max(torch.sqrt(torch.sum(pc_nonzeros**2, 1)))
    pc_nonzeros = pc_nonzeros / m
    pc[nonzeros_indices, :] = pc_nonzeros
    pc = pc.reshape(w, h, c)
    pc = pc.permute(2, 0, 1)
    return pc


class EyecandiesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split='test',
        random_seed=42,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        if split == DatasetSplit.TEST:
            self.split = 'test_public'
        elif split == DatasetSplit.TRAIN:
            self.split = 'train'
        elif split == DatasetSplit.VAL:
            self.split = 'val'
        self.source = source
        self.cls = classname
        self.img_path = os.path.join(source, self.cls, self.split, 'data')

        self.data_to_iterate = self.load_dataset()

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((resize,resize), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
            
        self.gt_transform = [
            transforms.Resize((resize,resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.gt_transform = transforms.Compose(self.gt_transform)

        self.imagesize = (3, imagesize, imagesize)

    
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        rgb_path = image_path[0]
        tiff_path = image_path[1]

        image = PIL.Image.open(rgb_path).convert("RGB")
        image = self.rgb_transform(image)

        organized_pc = read_tiff_organized_pc(tiff_path)
        # point cloud
        resized_organized_pc = resize_organized_pc(organized_pc, target_height=self.imagesize[-1], target_width=self.imagesize[-1])
        resized_organized_pc = pc_normalize_mvtec(resized_organized_pc)
        resized_organized_pc = resized_organized_pc.clone().detach().float()

        if mask_path == 0:
            mask = torch.zeros([1, resized_organized_pc.shape[-1], resized_organized_pc.shape[-1]])
        else:
            mask = Image.open(mask_path).convert('L')
            mask = self.gt_transform(mask)
            mask = torch.where(mask > 0.5, 1., .0)

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": anomaly,
            "image_path": rgb_path,
            "point_cloud": resized_organized_pc
        }

    def __len__(self):
        return len(self.data_to_iterate)
    
    def load_dataset(self):
        rgb_paths = glob.glob(self.img_path + "/*_image_0.png")
        tiff_paths = glob.glob(self.img_path + "/*_xyz.tiff")
        mask_paths = [tiff_path.replace('_xyz.tiff', '_mask.png') for tiff_path in tiff_paths]
        rgb_paths.sort()
        tiff_paths.sort()
        mask_paths.sort()
        gt_list = []
        for mask_path in mask_paths:
            image = np.array(PIL.Image.open(mask_path))
            if image.max() == 0:
                gt_list.append(0)
            else:
                gt_list.append(1)
        data_to_iterate = [(self.cls, gt_list[i], (rgb_paths[i], tiff_paths[i]), mask_paths[i]) for i in range(len(rgb_paths))]
        return data_to_iterate
