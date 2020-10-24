"""
Script for recover low resolution model predicted segmentation map to original HR space
It need the recorded txt info file for crop/pading and upsampling

Yucheng Tang
April 2019

"""


import os
import numpy as np
import nibabel as nb
from PIL import Image
import matplotlib.pyplot as plt

class Recoveror(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.checkpoint = args.checkpoint_dir
        self.segout_dir = args.segout_dir
        self.txt_info_dir = args.txt_info_dir

    def mkdir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def processing(self):
        print('Start recover to original space')
        cropped_image_dir = os.path.join(self.cropped_dir, 'soft_images')
        final_crop_file = os.path.join(self.txt_info_dir, 'resample_crop_file.txt')
        segout_dir = os.path.join(self.segout_dir, 'seg')
        output_segout_dir = os.path.join(self.segout_dir, 'high_seg')
        if not os.path.isdir(output_segout_dir):
            os.makedirs(output_segout_dir)
        count = 0 
        for image in os.listdir(segout_dir):
            count += 1
            image_path = os.path.join(segout_dir, image)
            # imgnb = nb.load(image_path)
            # imgnp = np.array(imgnb.dataobj

            with open(final_crop_file) as f:
                content = f.readlines()
            content = [x.strip() for x in content]

            image_name = image
            for item in content:
                # item = item.replace('_soft', '')
                if image_name in item:
                    item_name = item.split(' ')[0]
                    dim_x = int(item.split(' ')[1])
                    dim_y = int(item.split(' ')[2])
                    dim_z = int(item.split(' ')[3])

                    x_ori = int(float(item.split(' ')[4]))
                    y_ori = int(float(item.split(' ')[5]))
                    z_ori = int(float(item.split(' ')[6]))

                    pixdim_x = float(item.split(' ')[7])
                    pixdim_y = float(item.split(' ')[8])
                    pixdim_z = float(item.split(' ')[9])


                    x_start = 0 - int(x_ori)
                    y_start = 0 - int(y_ori)
                    z_start = 0 - int(z_ori)

                    # Rescale
                    os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                        image_path, image_path, x_start, dim_x, y_start, dim_y, z_start, dim_z
                    ))
                    print('[{}] Padding or crop to original scale {}'.format(count, item_name))
                    output_segout_file = os.path.join(output_segout_dir, image)
                    # Upsample
                    os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt nearest'.format(image_path, output_segout_file, pixdim_x, pixdim_y, pixdim_z))

                    # load seg dimension
                    output_segnb = nb.load(output_segout_file)
                    output_segnp = np.array(output_segnb.dataobj)
                    seg_dim_x = output_segnp.shape[0]
                    seg_dim_y = output_segnp.shape[1]
                    seg_dim_z = output_segnp.shape[2]

                    # load img dimension
                    cropped_image_path = os.path.join(cropped_image_dir, image)
                    cropped_imgnb = nb.load(cropped_image_path)
                    cropped_imgnp = np.array(cropped_imgnb.dataobj)
                    cropped_dim_x = cropped_imgnp.shape[0]
                    cropped_dim_y = cropped_imgnp.shape[1]
                    cropped_dim_z = cropped_imgnp.shape[2]

                    # Set the volume in the center, or crop from center
                    x_ori = (seg_dim_x - cropped_dim_x) / 2 # Floor operation 
                    y_ori = (seg_dim_y - cropped_dim_y) / 2
                    z_ori = (seg_dim_z - cropped_dim_z) / 2
                    
                    os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                        output_segout_file, output_segout_file, x_ori, 512, y_ori, 512, z_ori, cropped_dim_z
                    ))
                    fOutput_nb = nb.load(output_segout_file)
                    fOutput_np = np.array(fOutput_nb.dataobj)
                    image_file = os.path.join(cropped_image_dir, item_name)
                    imgnb = nb.load(image_file)
                    alignedOut = nb.Nifti1Image(fOutput_np, imgnb.affine)
                    nb.save(alignedOut, output_segout_file)
                    print('[{}] Fine tuning image {} from {} {} {} to 512 512 {}'.format(count, image, seg_dim_x, seg_dim_y, seg_dim_z, cropped_dim_z))


