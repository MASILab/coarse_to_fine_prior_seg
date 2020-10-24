import os
import numpy as np
from torch.utils import data
import nibabel as nib


output_x = 168
output_y = 168
output_z = 64



class pytorch_loader(data.Dataset):
    def __init__(self, subdict):
        self.subdict = subdict
        self.img_subs = subdict['img_subs']
        self.img_files = subdict['img_files']

    def __getitem__(self, index):

        labels = range(13)
        sub_name = self.img_subs[index]
        x = np.zeros((1, output_z, output_x, output_y))
        img_file = self.img_files[index]
        img_3d = nib.load(img_file)
        img = img_3d.get_data()
        img = (img - img.min())/(img.max()-img.min())
        img = img*255.0
        img = np.transpose(img,(2, 0, 1))
        x[0,:,:,:] = img[0:output_z,0:output_x,0:output_y]
        x = x.astype('float32')
        y = x

        return x, y, sub_name

    def __len__(self):
        return len(self.img_subs)
