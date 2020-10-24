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
import random



class Random_sampler(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.checkpoint = args.checkpoint_dir
        self.segout_dir = args.segout_dir
        self.txt_info_dir = args.txt_info_dir
        self.randPatch_dir = args.randPatch_dir
        # self.targetOrgan = args.targetOrgan

    def mkdir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def processing(self):
        for i in range(1, 14):
            self.targetOrgan = str(i)
            cropped_image_dir = os.path.join(self.cropped_dir, 'images')
            cropped_soft_image_dir = os.path.join(self.cropped_dir, 'soft_images')

            output_segout_dir = os.path.join(self.segout_dir, 'high_seg')

            segout_dir = os.path.join(self.segout_dir, 'seg')

            count = 0 

            index2organ = {1 : 'spleen', 2 : 'rk',  3 : 'lk', 4 : 'gall',  5 : 'eso',  6 : 'liver', \
                7 : 'stomach', 8 : 'aorta', 9 : 'IVC', 10 : 'PSV', 11 : 'pancreas', 12 : 'rad', 13 : 'lad'}

            organ2index = {'spleen' : 1, 'rk' : 2, 'lk' : 3, 'gall' : 4, 'eso' : 5, 'liver' : 6, \
                'stomach' : 7, 'aorta' : 8, 'IVC' : 9, 'PSV' : 10, 'pancreas' : 11, 'rad' : 12, 'lad' : 13}

            patch_size = [128,128,48]
            patch_number = 50      

            count = 0
            coor_file = os.path.join(self.txt_info_dir, 'patches_128_50_coor_{}.txt'.format(self.targetOrgan))
            coor_file_wr = open(coor_file, 'w')
            # missing_organ_list_file = os.path.join(coor_dir, 'random_patch_missingOrgan_lits.txt')
            # misingOrgan_file_wr = open(missing_organ_list_file, 'w')
            seg_patch_dir = os.path.join(self.randPatch_dir, 'seg_patches_{}'.format(self.targetOrgan))
            # img_patch_dir = os.path.join(self.randPatch_dir, 'img_patches_{}'.format(self.targetOrgan))
            soft_patch_dir = os.path.join(self.randPatch_dir, 'softimg_patches_{}'.format(self.targetOrgan))

            # if not os.path.isdir(img_patch_dir):
            #     os.makedirs(img_patch_dir)
            if not os.path.isdir(seg_patch_dir):
                os.makedirs(seg_patch_dir)
            if not os.path.isdir(soft_patch_dir):
                os.makedirs(soft_patch_dir)

            targetOrgan = int(self.targetOrgan)

            for seg in os.listdir(output_segout_dir):
                count += 1

                seg_file = os.path.join(output_segout_dir, seg)
                segnb = nb.load(seg_file)
                segnp = np.array(segnb.dataobj)

                # image_file = os.path.join(cropped_image_dir, seg)
                # imgnb = nb.load(image_file)
                # imgnp = np.array(imgnb.dataobj)

                soft_file = os.path.join(cropped_soft_image_dir, seg)
                softnb = nb.load(soft_file)
                softnp = np.array(softnb.dataobj)

                #create patch dirs

                # start croping
                coor = []
                organ_dimensions = []

                index_tuple = np.where(segnp == targetOrgan)
                # skip  subject with no target or any voxel prediction
                if len(index_tuple[0]) <= 50:
                    print('[{}] length: {}, image : {}, skipping'.format(count, len(index_tuple[0]), seg))
                    continue

                index_np = np.array(index_tuple)
                x_index_min = index_np[0].min()
                x_index_max = index_np[0].max()
                y_index_min = index_np[1].min()
                y_index_max = index_np[1].max()
                z_index_min = index_np[2].min()
                z_index_max = index_np[2].max()
                index_list = []
                for idx, voxel in enumerate(index_np[0]):
                    voxel_list = [index_np[0][idx], index_np[1][idx], index_np[2][idx]]
                    index_list.append(voxel_list)
                
                random.shuffle(index_list)
                sampled_voxel_list = random.sample(index_list, patch_number)
                # Get organ range from GT 
                for idx, item in enumerate(sampled_voxel_list): 
                    # make each label voxel the center coordinate
                    x_ori = item[0] - patch_size[0] / 2
                    y_ori = item[1] - patch_size[1] / 2
                    z_ori = item[2] - patch_size[2] / 2
                    # output_img_patch_file = os.path.join(img_patch_dir, '{}_{}_{}.nii.gz'.format(seg, \
                    #     index2organ[targetOrgan], idx))
                    output_seg_patch_file = os.path.join(seg_patch_dir, '{}_{}_{}.nii.gz'.format(seg, \
                        index2organ[targetOrgan], idx))
                    output_soft_patch_file = os.path.join(soft_patch_dir, '{}_{}_{}.nii.gz'.format(seg, \
                        index2organ[targetOrgan], idx))

                    # os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                    #     image_file, output_img_patch_file, x_ori, patch_size[0], y_ori, patch_size[1],\
                    #     z_ori, patch_size[2]
                    # ))        
                    os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                        seg_file, output_seg_patch_file, x_ori, patch_size[0], y_ori, patch_size[1],\
                        z_ori, patch_size[2]
                    ))  
                    os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                        soft_file, output_soft_patch_file, x_ori, patch_size[0], y_ori, patch_size[1],\
                        z_ori, patch_size[2]
                    ))                 
                    print('[{} -- {}---{}] soft image and seg patches saved {}'.format(count, \
                        index2organ[targetOrgan], idx, seg))
                    # note voxel coordinate down
                    coor_file_wr.write(seg + ' ' + 'patch_number:{}'.format(idx) + ' ' + str(item[0]) + \
                        ' ' + str(item[1]) + ' ' + str(item[2]) + '\n')

            print('[{}] Image crop coordinated saved {}'.format(count, seg))
            coor_file_wr.close()

        # clean labels 

            count = 0
            for seg in os.listdir(seg_patch_dir):
                count += 1
                seg_file = os.path.join(seg_patch_dir, seg)
                segnb = nb.load(seg_file)
                segnp = np.array(segnb.dataobj)

                # organ_label = organ2index[organ]
                idx = np.where(segnp != targetOrgan)
                segnp[idx] = 0

                idx = np.where(segnp != 0)
                segnp[idx] = 1

                lablenb_new = nb.Nifti1Image(segnp, segnb.affine)

                label_newfile = os.path.join(seg_patch_dir, seg)
                nb.save(lablenb_new, label_newfile)
                print('[{}] Converting {}, to 0, then label to 1'.format(count, seg)) 






#

