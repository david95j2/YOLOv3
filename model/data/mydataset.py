import numpy as np
import torch
import torchvision
from skimage import transform as sk_transform 

# from Data pacakage
from utils.config import opt
import data_util
from .custom_dataset import MyBboxDataset


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
        
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    # appr -1~1 RGB
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))

    return img.numpy()


def caffe_normalize(img):
    # appr -125-125 BGR
    
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255

    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)

    return img


def preprocess(img, min_size=600, max_size=1000):
    
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    
    img = img / 255.
    img = sk_transform.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze

    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = data_util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = data_util.random_flip(img, x_random=True, return_param=True)
        bbox = data_util.flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = MyBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='val', use_difficult=True):
        self.opt = opt
        self.db = MyBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)