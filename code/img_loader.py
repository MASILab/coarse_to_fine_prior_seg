import os
import h5py
import numpy as np
from torch.utils import data
import os.path
import torchvision.transforms as transforms
from PIL import Image
import torch
import random_crop_yh

VGG_MEAN = [103.939, 116.779, 123.68]


def normalizeImage(img):
    img = img.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = img.min()
        maxval = img.max()
        if minval != maxval:
            img -= minval
            img /= (maxval-minval)
    return img*255


class img_loader(data.Dataset):
    def __init__(self, sub_list):
        self.sub_list = sub_list

        self.finesize = [128, 128]




        transform_list = []
        transform_list.append(transforms.Scale(self.finesize, Image.BICUBIC))
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(self.finesize, Image.NEAREST))
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_crop_yh.randomcrop_yh(self.finesize))
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5],[0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # load image
        subinfo = self.sub_list[index]

        image_name = subinfo[0]
        image_path = subinfo[1]

        A_img = Image.open(image_path)
        # Seg_img = Image.open(seg_path).convert('I')


        A_img = self.transforms_scale(A_img)
        # Seg_img = self.transforms_seg_scale(Seg_img)

        A_img = self.transforms_toTensor(A_img)
        # Seg_img = self.transforms_toTensor(Seg_img)

        A_img = self.transforms_normalize(A_img)
        data = torch.Tensor(1, self.finesize[0], self.finesize[1])
        # data[2, :, :] = A_img
        # data[1, :, :] = A_img
        data[0, :, :] = A_img

        # Seg_imgs = torch.Tensor(2, 512, 512)
        # Seg_imgs[0, :, :] = Seg_img[0] == 0
        # Seg_imgs[1, :, :] = Seg_img[0] != 1
        return data, image_name




    def __len__(self):
        self.total_count = len(self.sub_list)
        return self.total_count


    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += np.array(VGG_MEAN)
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl











