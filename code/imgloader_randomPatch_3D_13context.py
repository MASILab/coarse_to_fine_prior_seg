import os
import numpy as np
from torch.utils import data
import nibabel as nib


output_x = 128
output_y = 128
output_z = 48

# labels = [0, 4,11,23,30,31,32,35,36,37,38,39,40,41,44,45,47,48,49,50,51,52,55,56,57,58,59,60,61,62,71,72,73,75,76,100,101,102,103,104,105,106,107,108,109,112,113,114,115,116,117,118,119,120,121,122,123,124,125,128,129,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207]
class pytorch_loader(data.Dataset):
    def __init__(self, subdict, num_labels):
        self.subdict = subdict
        self.img_subs = subdict['img_subs']
        self.img_files = subdict['img_files']

        self.pred_subs = subdict['pred_subs']
        self.pred_files = subdict['pred_files']

        if 'seg_subs' in subdict:
            self.seg_subs = subdict['seg_subs']
            self.seg_files = subdict['seg_files']
        else:
            self.seg_subs = None
            self.seg_files = None
            self.seg_resource = None
        self.num_labels = num_labels

    def __getitem__(self, index):

        num_labels = self.num_labels
        labels = range(num_labels)
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
        # load pred segmentations
        pred_file = self.pred_files[index]
        pred_3d = nib.load(pred_file)
        pred = pred_3d.get_data()
        pred = np.transpose(pred,(2, 0, 1))
        a = np.unique(pred)
        z = np.zeros((13, output_z, output_x, output_y))
        for i in range(0,13):
            if i in a:
                pred_one = pred == i
                z[i,:,:,:] = pred_one[0:output_z,0:output_x,0:output_y]
                z = z.astype('float32')
            else:
                zero_m = np.zeros((1, output_z, output_x, output_y))
                z[i,:,:,:] = zero_m
        # print(z.shape)
        z[z!=0] = 1
        
        if (self.seg_files == None):
            y = x
        else:
            y = np.zeros((num_labels, output_z, output_x, output_y))
            seg_file = self.seg_files[index]
            seg_3d = nib.load(seg_file)
            seg = seg_3d.get_data()
            seg = np.transpose(seg,(2, 0, 1))
            y[0,:,:,:] = np.ones([output_z,output_x,output_y])
            for i in range(1,num_labels):
                seg_one = seg == labels[i]
                y[i,:,:,:] = seg_one[0:output_z,0:output_x,0:output_y]
                y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
                y = y.astype('float32')
            # organ_idx = sub_name.split('.nii.gz_')[1].split('.nii.gz')[0].split('_')[1]
        
        input_len = 14
        o = np.zeros((input_len, output_z, output_x, output_y))
        for i in range(input_len):
            if i == 0:
                o[i] = x
            elif i == input_len - 1:
                o[i] = z[i-1,:,:,:]
                break
            else:
                o[i] = z[i,:,:,:]
        o = o.astype('float32')

        return o, y, sub_name

    def __len__(self):
        return len(self.img_subs)