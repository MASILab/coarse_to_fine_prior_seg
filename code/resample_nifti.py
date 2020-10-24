"""
Script for preprocessing of training purpose
mode1: same spacing but may contains many 0 planes
mode2: keep dimension consistant as much as possible, without same spacing.

mode1: images: 1) Normalization, 2) downsample with cubic spline interpolation 3) Padding or crop
       labels: 1) nearest interpolation 2) padding or crop
       images soft: 1) down sample cubic interpolation 2) pad or crop
mode2: images: 1) Normalization 2) downsample to given dimension(no consistant spacing) 3) pad or crop
       labels: 1) nearest interpolation 2)pad crop
       images soft: 1) down sample cubic interpolation 2) pad or crop

Yucheng Tang
Oct 2018

"""

import os
import numpy as np
import nibabel as nb
from nibabel.processing import resample_from_to 
from PIL import Image
import matplotlib.pyplot as plt

class Resample_nifti(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.txt_info_dir = args.txt_info_dir

        self.dim = []
        self.dim.append(int(args.dim.split(' ')[0]))
        self.dim.append(int(args.dim.split(' ')[1]))
        self.dim.append(int(args.dim.split(' ')[2]))

        self.pixdim = []
        self.pixdim.append(float(args.pixdim.split(' ')[0]))
        self.pixdim.append(float(args.pixdim.split(' ')[1]))
        self.pixdim.append(float(args.pixdim.split(' ')[2]))

        self.score_interval = []
        self.score_interval.append(float(args.score_interval.split(' ')[0]))
        self.score_interval.append(float(args.score_interval.split(' ')[1]))

    def processing(self):
        output_image_dir = os.path.join(self.output_dir, 'images')
        output_image_soft_dir = os.path.join(self.output_dir, 'soft_images')
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
        if not os.path.isdir(output_image_soft_dir):
            os.makedirs(output_image_soft_dir)
        finesize = self.dim


        final_crop_file = os.path.join(self.txt_info_dir, 'resample_crop_file.txt')
        final_crop_info_file = open(final_crop_file, 'w')
        # 1: Resample to consistent spcaing(pixel dimension) ----Image
        # count = 0
        # cropped_image_dir = os.path.join(self.cropped_dir, 'images')
        # for image in os.listdir(cropped_image_dir):
        #     try:
        #         image_path = os.path.join(cropped_image_dir, image)
        #         imgnb = nb.load(image_path)
        #         imgnp = np.array(imgnb.dataobj)
        #     except IOError:
        #         print('IO Error, Skip {}'.format(image))
        #         continue
        #     #4.1 Normalization 
        #     count += 1
        #     idx = np.where(imgnp < -1000)
        #     imgnp[idx[0], idx[1], idx[2]] = -1000 # set minmum to -175
        #     idx = np.where(imgnp > 1000)
        #     imgnp[idx[0], idx[1], idx[2]] = 1000 # set maximum to 275
        #     imgnp = (imgnp - imgnp.min()) * (1.0 - 0.0) / (imgnp.max() - imgnp.min())
        #     imgnb = nb.Nifti1Image(imgnp, imgnb.affine)
        #     print('[{}] Normalizing image {} to [0,1]'.format(count, image))
        #     #4.2 Resample the image to certain space using spline interpolation
        #     output_image_file = os.path.join(output_image_dir, image)   
        #     nb.save(imgnb, output_image_file) 
        #     os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(output_image_file, output_image_file, self.pixdim[0], self.pixdim[1], self.pixdim[2]))
        

        # 2: Resample Soft Tissue Images to certain space using interpolation
        count = 0
        cropped_soft_dir = os.path.join(self.cropped_dir, 'soft_images')
        for image_soft in os.listdir(cropped_soft_dir):
            count += 1
            image_soft_path = os.path.join(cropped_soft_dir, image_soft)
            output_soft_file = os.path.join(output_image_soft_dir, image_soft)
            if os.path.isfile(output_soft_file):
                print('[{}] Exits, continue {}'.format(count, image_soft))
                continue
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(image_soft_path, output_soft_file, self.pixdim[0], self.pixdim[1], self.pixdim[2]))


        #4.4 Padding or crop to certain size 
        # print('Fine tune all volumes to trainable size')
        # count = 0
        # for image in os.listdir(output_image_dir):
        #     # Image 
        #     count += 1
        #     image_path = os.path.join(output_image_dir, image)
        #     imgnb = nb.load(image_path)
        #     dim_x = imgnb.shape[0]
        #     dim_y = imgnb.shape[1]
        #     dim_z = imgnb.shape[2]

        #     # Set the volume in the center, or crop from center
        #     x_ori = (dim_x - self.dim[0]) / 2 # Floor operation 
        #     y_ori = (dim_y - self.dim[1]) / 2
        #     z_ori = (dim_z - self.dim[2]) / 2
            
        #     # os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
        #     #     image_path, image_path, x_ori, finesize[0], y_ori, finesize[1], z_ori, finesize[2]
        #     # ))
        #     os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
        #         image_path, image_path, x_ori, finesize[0], y_ori, finesize[1], z_ori, finesize[2]
        #     ))
        #     print('[{}] Fine tuning image {} from {} {} {} to {}'.format(count, image, dim_x, dim_y, dim_z, str(self.dim)))

        print('Fine tune all soft volumes to trainable size')
        count = 0
        for image_soft in os.listdir(output_image_soft_dir):
            # Image 
            count += 1
            image_soft_path = os.path.join(output_image_soft_dir, image_soft)
            imgnb = nb.load(image_soft_path)
            dim_x = imgnb.shape[0]
            dim_y = imgnb.shape[1]
            dim_z = imgnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.dim[1]) / 2
            z_ori = (dim_z - self.dim[2]) / 2

            # keep note 
            original_nib_path = os.path.join(self.cropped_dir, 'soft_images', image_soft)
            if not os.path.isfile(original_nib_path):
                print('Error! File not Found')
                continue
            original_nib = nb.load(original_nib_path)
                        
            final_crop_info_file.write(image_soft + ' ' + str(dim_x) + ' ' + str(dim_y) + ' ' + str(dim_z) + ' ' + str(x_ori) + ' ' + str(y_ori) + ' ' + str(z_ori) + ' ' + str(original_nib.header['pixdim'][1]) + ' ' + str(original_nib.header['pixdim'][2]) + ' ' + str(original_nib.header['pixdim'][3]) + '\n')

            # os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
            #     image_soft_path, image_soft_path, x_ori, finesize[0], y_ori, finesize[1], z_ori, finesize[2]
            # ))
            os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                image_soft_path, image_soft_path, x_ori, finesize[0], y_ori, finesize[1], z_ori, finesize[2]
            ))
            print('[{}] Fine tuning soft image {} from {} {} {} to {}'.format(count, image_soft, dim_x, dim_y, dim_z, str(self.dim)))
        






      # Processing labels
        # count = 0
        # cropped_label_dir = os.path.join(self.cropped_dir, 'labels')
        # output_label_dir = os.path.join(self.output_dir, 'labels')
        # if not os.path.isdir(output_label_dir):
        #     os.makedirs(output_label_dir)
        # for label in os.listdir(cropped_label_dir):
        #     count += 1
        #     label_path = os.path.join(cropped_label_dir, label)
        #     output_label_file = os.path.join(output_label_dir, label)
        #     if os.path.isfile(output_label_dir):
        #         print('[{}] Exits, continue {}'.format(count, label))
        #         continue
        #     os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt nearest'.format(label_path, output_label_file, self.pixdim[0], self.pixdim[1], self.pixdim[2]))



        # print('Fine tune labels to trainable size')
        # count = 0
        # for label in os.listdir(output_label_dir):
        #     # Image 
        #     count += 1
        #     label_path = os.path.join(output_label_dir, label)
        #     imgnb = nb.load(label_path)
        #     dim_x = imgnb.shape[0]
        #     dim_y = imgnb.shape[1]
        #     dim_z = imgnb.shape[2]

        #     # Set the volume in the center, or crop from center
        #     x_ori = (dim_x - self.dim[0]) / 2 # Floor operation 
        #     y_ori = (dim_y - self.dim[1]) / 2
        #     z_ori = (dim_z - self.dim[2]) / 2
            
        #     os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
        #         label_path, label_path, x_ori, finesize[0], y_ori, finesize[1], z_ori, finesize[2]
        #     ))
        #     print('[{}] Fine tuning label {} from {} {} {} to {}'.format(count, label, dim_x, dim_y, dim_z, str(self.dim)))
        


        final_crop_info_file.close()