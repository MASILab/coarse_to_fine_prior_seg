"""
Script for croping training images from Ground Truth segmentation map
Generate patches, High Resolution

Yucheng Tang
April 2019
"""

import os
import numpy as np
import nibabel as nb
from nibabel.processing import resample_from_to 
from PIL import Image
import matplotlib.pyplot as plt

class CropHRPatch(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.data_dir = args.data_dir
        self.caffe_dir = args.caffe_dir
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.txt_info_dir = args.txt_info_dir


        ### organ distribution range
        self.index2organ = {1 : 'spleen', 2 : 'rk',  3 : 'lk', 4 : 'gall',  5 : 'eso',  6 : 'liver', \
            7 : 'stomach', 8 : 'aorta', 9 : 'IVC', 10 : 'PSV', 11 : 'pancreas', 12 : 'ad'}
        self.organ2patches = {'spleen' : [2,2,1], 'rk': [2,2,1], 'lk' : [2,2,1], 'gall': [1,1,1],\
            'eso' : [1,1,1], 'liver': [3,2,2], 'stomach' : [2,2,2], 'aorta' : [1,1,2], \
            'IVC' : [1,1,2], 'PSV' : [3,2,1], 'pancreas': [2,2,1], 'ad' : [2,1,1]}
        self.patch_size = [112,112,48]

    def processing(self):
        imgdir = os.path.join(self.output_dir, 'HR_images')
        output_HR_image_soft_dir = os.path.join(self.output_dir, 'HR_soft_images')



            





segdir = ''
imgdir = ''
coor_dir = ''
patch_dir = ''

count = 0
coor_file = os.path.join(coor_dir, 'crop_coor_list.txt')
coor_file_wr = open(coor_file, 'w')
for seg in os.listdir(segdir):
    count += 1
    seg_file = os.path.join(segdir, seg)
    segnb = nb.load(seg_file)
    segnp = np.array(segnb.dataobj)
    img_file = os.path.join(imgdir, seg)
    imgnb = nb.load(img_file)
    imgnp = np.array(imgnb.dataobj)
    #create patch dirs
    img_patch_dir = os.path.join(patch_dir, 'image_patches')
    seg_patch_dir = os.path.join(patch_dir, 'seg_patches')
    if not os.path.isdir(img_patch_dir):
        os.makedirs(img_patch_dir)
    if not os.path.isdir(seg_patch_dir):
        os.makedirs(seg_patch_dir)

    imgShape = [imgnp.shape[0], imgnp.shape[1], imgnp.shape[2]] 


    coor = []
    organ_dimensions = []
    for i in range(1, 13):
        index_tuple = np.where(segnp == i)
        index_np = np.array(index_tuple)
        x_index_min = index_np[0].min()
        x_index_max = index_np[0].max()
        y_index_min = index_np[1].min()
        y_index_max = index_np[1].max()
        z_index_min = index_np[2].min()
        z_index_max = index_np[2].max()

        # Get organ range from GT 
        organ_range_LT = [x_index_min, y_index_min, z_index_min]
        organ_range_RB = [x_index_max, y_index_max, z_index_max]

        organ_dimension = [x_index_max-x_index_min, y_index_max-y_index_min, z_index_max-z_index_min]
        organ_dimensions.append(organ_dimension)
        
        patch_number = organ2patches[index2organ[i]]
        # Determin organ patch size
        organ_patch_size = [patch_size[0] * patch_number[0], patch_size[1] * patch_number[1],\
             patch_size[2] * patch_number[2]]
        # Calculate crop starting top left point and right bottom point
        ori_coor = []
        end_coor = []
        for j in range(3):
            current_ori_coor = organ_range_LT[j] - int((organ_dimension[j] - organ_patch_size[j]) / 2)
            current_end_coor = current_ori_coor + organ_patch_size[j]
            #floor operation
            # If box coordinates exceed the bondary, move to the border
            if current_ori_coor < 0:
                number_mv = 0 - current_ori_coor
                current_ori_coor = 0
                current_end_coor += number_mv
            elif current_end_coor > imgShape[j]:
                number_mv = current_end_coor - imgShape[j] + 1
                current_end_coor = imgShape[j] - 1
                current_ori_coor -= number_mv
                
            ori_coor.append(current_ori_coor)
            end_coor.append(current_end_coor)

        # Crop oprgan here
        img_organ_patch = imgnp[ori_coor[0]:end_coor[0],ori_coor[1]:end_coor[1],ori_coor[2]:end_coor[2]]
        seg_organ_patch = segnp[ori_coor[0]:end_coor[0],ori_coor[1]:end_coor[1],ori_coor[2]:end_coor[2]]
        
        # Split organ patch to trainable size [112,112, 48]
        xyz_nums = [int(organ_patch_size[0] / patch_size[0]), \
            int(organ_patch_size[1] / patch_size[1]), int(organ_patch_size[2] / patch_size[2])]
        all_patches_ori_coor = []
        all_x_ori_coor = []
        all_y_ori_coor = []
        all_z_ori_coor = []
        for k in range(xyz_nums[0]):
            all_x_ori_coor.append(k * patch_size[0])
        for k in range(xyz_nums[1]):
            all_y_ori_coor.append(k * patch_size[1])
        for k in range(xyz_nums[2]):
            all_z_ori_coor.append(k * patch_size[2])
        # For croping, we define indices of patches from x, y and z direction
        for z_coor in all_z_ori_coor:
            for y_coor in all_y_ori_coor:
                for x_coor in all_x_ori_coor:
                    all_patches_ori_coor.append([x_coor, y_coor, z_coor]) 
        for patch_idx in range(xyz_nums[0] * xyz_nums[1] * xyz_nums[2]):
            subpatch_coor = all_patches_ori_coor[patch_idx]
            img_organ_subpatch = img_organ_patch[subpatch_coor[0] : subpatch_coor[0] + patch_size[0],\
                subpatch_coor[1] : subpatch_coor[1] + patch_size[1],\
                subpatch_coor[2] : subpatch_coor[2] + patch_size[2]]

            seg_organ_subpatch = seg_organ_patch[subpatch_coor[0] : subpatch_coor[0] + patch_size[0],\
                subpatch_coor[1] : subpatch_coor[1] + patch_size[1],\
                subpatch_coor[2] : subpatch_coor[2] + patch_size[2]]

            affine_mx = np.array([[imgnb.affine[0][0], 0, 0, 0], [0, imgnb.affine[1][1], 0, 0],\
                [0, 0, imgnb.affine[2][2], 0], [0,0,0,1]])

            img_subpatchnb = nb.Nifti1Image(img_organ_subpatch, affine_mx)
            seg_subpatchnb = nb.Nifti1Image(seg_organ_subpatch, affine_mx)

            subpatch_name = seg + '_' + index2organ[i] + '_' + str(patch_idx) + '.nii.gz'
            img2organ_patch_file = os.path.join(img_patch_dir, subpatch_name)
            seg2organ_patch_file = os.path.join(seg_patch_dir, subpatch_name)
            nb.save(img_subpatchnb, img2organ_patch_file) 
            nb.save(seg_subpatchnb, seg2organ_patch_file) 
            print('[{} -- {} -- {}] image and seg patches saved {}'.format(count, i, patch_idx, seg))
 


        #Save crop coordinates information
        final_coor = []
        final_coor.append(ori_coor)
        final_coor.append(end_coor)
        coor.append(final_coor)
    coor_file_wr.write(seg + ' ' + str(coor) + '\n')
    print('[{}] Image crop coordinated saved {}'.format(count, seg))
coor_file_wr.close()