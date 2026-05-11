import torch
from torchvision import transforms
from PIL import ImageFilter
import tifffile as tiff
import numpy as np

class KNNGaussianBlur(torch.nn.Module):
    def __init__(self, radius : int = 4):
        super().__init__()
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=radius)

    def __call__(self, img):
        img = torch.tensor(img).squeeze()
        map_max = img.max()
        final_map = self.load(self.unload(img / map_max).filter(self.blur_kernel)) * map_max
        final_map = np.array(final_map.permute(1,2,0))
        return final_map


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width), mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]