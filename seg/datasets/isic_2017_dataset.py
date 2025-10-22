import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import DataLoader
import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ('Target', 'Background')
PALETTE = [[255], [0]]

def get_training_transform():
    train_transform = [
        albu.Resize(height=512, width=512, p=1.0),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    #crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5,1.75], mode='value'),
    #                    SmartCropV1(crop_size=256, max_ratio=0.75,ignore_index=len(CLASSES), nopad=False)])
    #img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Resize(height=512, width=512, p=1.0),
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class ISIC2017Dataset(Dataset):
    def __init__(self, data_root='/data/cj/dataset/2dv/m/isic/isic2017/', mode='val',
                  transform=val_aug, mosaic_ratio=0.0):
        self.data_root = data_root
        self.sample_lists = []
        self.mode = mode
        
        if mode == "test":
            mode = "val"
        image_dir = os.path.join(data_root, mode, "images")
        mask_dir = os.path.join(data_root, mode, "masks")
       
        for img_name in os.listdir(image_dir):
            index,png = img_name.split(".") # ISIC_0000000.jpg 
            mask_name = index + "_segmentation.png" # ISIC_0000000_segmentation.png
            self.sample_lists.append([os.path.join(image_dir,img_name),os.path.join(mask_dir, mask_name)])
        
        self.transform = transform
        
        self.mosaic_ratio = mosaic_ratio

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2,0,1).float()
        mask = torch.from_numpy(mask).long()
        
        if self.mode == "test":
            return img, mask, self.sample_lists[index][0].split(".")[-2].split("/")[-1]
        else:
            return img,mask

    def __len__(self):
        return len(self.sample_lists)


    def load_img_and_mask(self, index):
        img_name = self.sample_lists[index][0]
        ann_name = self.sample_lists[index][1]
        
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(ann_name).convert('L')       
        mask = np.array(mask)
        mask [mask > 0 ] = 1
        mask = Image.fromarray(mask)
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask