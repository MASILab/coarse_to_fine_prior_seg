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

class Resample_nifti_HR(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.txt_info_dir = args.txt_info_dir

        self.HR_dim = []
        self.HR_dim.append(int(args.HR_dim.split(' ')[0]))
        self.HR_dim.append(int(args.HR_dim.split(' ')[1]))
        self.HR_dim.append(int(args.HR_dim.split(' ')[2]))

        self.HR_pixdim = []
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[0]))
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[1]))
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[2]))

    def processing(self):
        output_HR_data_dir = os.path.join(self.output_dir, 'HR_data')
        if not os.path.isdir(output_HR_data_dir):
            os.makedirs(output_HR_data_dir)
        output_HR_image_dir = os.path.join(output_HR_data_dir, 'HR_images')
        output_HR_soft_image_dir = os.path.join(output_HR_data_dir, 'HR_soft_images')
        output_HR_seg_dir = os.path.join(output_HR_data_dir, 'HR_segs')
        output_HR_pred_dir = os.path.join(output_HR_data_dir, 'HR_preds')

        if not os.path.isdir(output_HR_image_dir):
            os.makedirs(output_HR_image_dir)
        if not os.path.isdir(output_HR_soft_image_dir):
            os.makedirs(output_HR_soft_image_dir)
        if not os.path.isdir(output_HR_seg_dir):
            os.makedirs(output_HR_seg_dir)
        if not os.path.isdir(output_HR_pred_dir):
            os.makedirs(output_HR_pred_dir)

        finesize = self.HR_dim


        HR_final_crop_file = os.path.join(self.txt_info_dir, 'HR_resample_crop_file.txt')
        HR_final_crop_info_file = open(HR_final_crop_file, 'w')
        # 1: Resample to consistent spcaing(pixel dimension) ----Image
        count = 0
        cropped_image_dir = os.path.join(self.cropped_dir, 'images')
        for image in os.listdir(cropped_image_dir):
            try:
                image_path = os.path.join(cropped_image_dir, image)
                imgnb = nb.load(image_path)
                imgnp = np.array(imgnb.dataobj)
            except IOError:
                print('IO Error, Skip {}'.format(image))
                continue
            #4.1 Normalization 
            count += 1
            imgnp = (imgnp - imgnp.min()) * (1.0 - 0.0) / (imgnp.max() - imgnp.min())
            imgnb = nb.Nifti1Image(imgnp, imgnb.affine)
            print('[{}] Normalizing image {} to [0,1]'.format(count, image))
            #4.2 Resample the image to certain space using spline interpolation
            output_image_file = os.path.join(output_HR_image_dir, image)   
            if os.path.isfile(output_image_file):
                print('[{}] Exits, continue {}'.format(count, image))
                continue
            nb.save(imgnb, output_image_file) 
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(output_image_file, output_image_file, self.HR_pixdim[0], self.HR_pixdim[1], self.HR_pixdim[2]))
        

        # 2: Resample Soft Tissue Images to certain space using interpolation
        count = 0
        cropped_soft_dir = os.path.join(self.cropped_dir, 'soft_images')
        for image_soft in os.listdir(cropped_soft_dir):
            count += 1
            image_soft_path = os.path.join(cropped_soft_dir, image_soft)
            output_soft_file = os.path.join(output_HR_soft_image_dir, image_soft)
            if os.path.isfile(output_soft_file):
                print('[{}] Exits, continue {}'.format(count, image_soft))
                continue
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt cubic'.format(image_soft_path, output_soft_file, self.HR_pixdim[0], self.HR_pixdim[1], self.HR_pixdim[2]))


      # 3: Processing labels
        count = 0
        cropped_label_dir = os.path.join(self.cropped_dir, 'labels')
        for label in os.listdir(cropped_label_dir):
            count += 1
            label_path = os.path.join(cropped_label_dir, label)
            output_label_file = os.path.join(output_HR_seg_dir, label)
            if os.path.isfile(output_label_file):
                print('[{}] Exits, continue {}'.format(count, label))
                continue
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt nearest'.format(label_path, output_label_file, self.HR_pixdim[0], self.HR_pixdim[1], self.HR_pixdim[2]))

      # 4: Processing preds
        count = 0
        cropped_pred_dir = os.path.join(self.output_dir, 'segout', 'high_seg')
        for pred in os.listdir(cropped_pred_dir):
            count += 1
            pred_path = os.path.join(cropped_pred_dir, pred)
            output_pred_file = os.path.join(output_HR_pred_dir, pred)
            if os.path.isfile(output_pred_file):
                print('[{}] Exits, continue {}'.format(count, pred))
                continue
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt nearest'.format(pred_path, output_pred_file, self.HR_pixdim[0], self.HR_pixdim[1], self.HR_pixdim[2]))


        #1 Padding or crop to certain size 
        print('Fine tune all volumes to trainable size')
        count = 0
        for image in os.listdir(output_HR_image_dir):
            # Image 
            count += 1
            image_path = os.path.join(output_HR_image_dir, image)
            imgnb = nb.load(image_path)
            dim_x = imgnb.shape[0]
            dim_y = imgnb.shape[1]
            dim_z = imgnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.HR_dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.HR_dim[1]) / 2
            # z_ori = (dim_z - self.HR_dim[2]) / 2
            
            os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                image_path, image_path, x_ori, finesize[0], y_ori, finesize[1], 0, dim_z
            ))
            print('[{}] Fine tuning image {} from {} {} {} to {}'.format(count, image, dim_x, dim_y, dim_z, str(self.HR_dim)))

    #     print('Fine tune all soft volumes to trainable size')
        count = 0
        for image_soft in os.listdir(output_HR_soft_image_dir):
            # Image 
            count += 1
            image_soft_path = os.path.join(output_HR_soft_image_dir, image_soft)
            imgnb = nb.load(image_soft_path)
            dim_x = imgnb.shape[0]
            dim_y = imgnb.shape[1]
            dim_z = imgnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.HR_dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.HR_dim[1]) / 2
            # z_ori = (dim_z - self.HR_dim[2]) / 2

            # keep note 
            original_nib_path = os.path.join(self.cropped_dir, 'soft_images', image_soft)
            if not os.path.isfile(original_nib_path):
                print('Error! File not Found')
                continue
            original_nib = nb.load(original_nib_path)
                        
            HR_final_crop_info_file.write(image_soft + ' ' + str(dim_x) + ' ' + str(dim_y) + ' ' + str(dim_z) + ' ' + str(x_ori) + ' ' + str(y_ori) + ' ' + str(0) + ' ' + str(original_nib.header['pixdim'][1]) + ' ' + str(original_nib.header['pixdim'][2]) + ' ' + str(original_nib.header['pixdim'][3]) + '\n')

            os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                image_soft_path, image_soft_path, x_ori, finesize[0], y_ori, finesize[1], 0, dim_z
            ))
            print('[{}] Fine tuning soft image {} from {} {} {} to {}'.format(count, image_soft, dim_x, dim_y, dim_z, str(self.HR_dim)))
        

        print('Fine tune labels to trainable size')
        count = 0
        for label in os.listdir(output_HR_seg_dir):
            # Image 
            count += 1
            label_path = os.path.join(output_HR_seg_dir, label)
            imgnb = nb.load(label_path)
            dim_x = imgnb.shape[0]
            dim_y = imgnb.shape[1]
            dim_z = imgnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.HR_dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.HR_dim[1]) / 2
            # z_ori = (dim_z - self.HR_dim[2]) / 2
            
            os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                label_path, label_path, x_ori, finesize[0], y_ori, finesize[1], 0, dim_z
            ))
            print('[{}] Fine tuning label {} from {} {} {} to {}'.format(count, label, dim_x, dim_y, dim_z, str(self.HR_dim)))
        
        print('Fine tune preds to trainable size')
        count = 0
        for pred in os.listdir(output_HR_pred_dir):
            # Image 
            count += 1
            pred_path = os.path.join(output_HR_pred_dir, pred)
            imgnb = nb.load(pred_path)
            dim_x = imgnb.shape[0]
            dim_y = imgnb.shape[1]
            dim_z = imgnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.HR_dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.HR_dim[1]) / 2
            # z_ori = (dim_z - self.HR_dim[2]) / 2
            
            os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                pred_path, pred_path, x_ori, finesize[0], y_ori, finesize[1], 0, dim_z
            ))
            print('[{}] Fine tuning pred {} from {} {} {} to {}'.format(count, pred, dim_x, dim_y, dim_z, str(self.HR_dim)))
    

        HR_final_crop_info_file.close()