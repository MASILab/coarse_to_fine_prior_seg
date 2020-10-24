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
April 2019
"""

import os
import numpy as np
import nibabel as nb
from nibabel.processing import resample_from_to 
from PIL import Image
import matplotlib.pyplot as plt

class PreSegs(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.data_dir = args.data_dir
        self.caffe_dir = args.caffe_dir
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.txt_info_dir = args.txt_info_dir
        self.GTsegs_dir = args.GTsegs_dir


        self.HR_dim = []
        self.HR_dim.append(int(args.HR_dim.split(' ')[0]))
        self.HR_dim.append(int(args.HR_dim.split(' ')[1]))

        self.HR_pixdim = []
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[0]))
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[1]))
        self.HR_pixdim.append(float(args.HR_pixdim.split(' ')[2]))


    def processing(self):
        # BPR implemented crop
        cropped_seg_dir = os.path.join(self.cropped_dir, 'segs')
        crop_index_txt_file = os.path.join(self.txt_info_dir, 'crop_index.txt')
        # with open(crop_index_txt_file) as f:
        #     content = f.readlines()
        # content = [x.strip() for x in content]
        # count = 0
        # for item in content:
        #     count += 1
        #     image_name = item.split(' ')[0]
        #     image_name = image_name.replace('_soft', '')
        #     z_min = int(item.split(' ')[1])
        #     z_max = int(item.split(' ')[2])
        #     # Manipulate directories
        #     seg_file = os.path.join(self.GTsegs_dir, image_name)
        #     segnb = nb.load(seg_file)
        #     segnp = np.array(segnb.dataobj)
        #     cropped_seg_dir = os.path.join(self.cropped_dir, 'segs')
        #     if not os.path.isdir(cropped_seg_dir):
        #         os.makedirs(cropped_seg_dir)
        #     out_cropped_segfile = os.path.join(cropped_seg_dir, image_name)
        #     # Implement crop here
        #     x_index = segnp.shape[0]
        #     y_index = segnp.shape[1]
        #     z_index = segnp.shape[2]

        #     if os.path.isfile(out_cropped_segfile):
        #         print('Skipping {}'.format(image_name))
        #         continue

        #     print('[{}] Cropping soft segmentation map {} from z index [{},{}] to [{},{}]'.format(count, image_name, 0, z_index,\
        #         z_min, z_max ))

        #     os.system('fsl5.0-fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(seg_file,out_cropped_segfile, 0, x_index, 0, y_index, int(z_min), int(z_max-z_min+1)))
        
        #  Resample to consistent spcaing(pixel dimension)
        out_HR_seg_dir = os.path.join(self.output_dir, 'HR_segs')
        if not os.path.isdir(out_HR_seg_dir):
            os.makedirs(out_HR_seg_dir)
        finesize = self.HR_dim
        for seg in os.listdir(cropped_seg_dir):
            try:
                seg_path = os.path.join(cropped_seg_dir, seg)
                segnb = nb.load(seg_path)
                segnp = np.array(segnb.dataobj)
            except IOError:
                print('IO Error, Skip {}'.format(seg))
                continue
            #4.1 Normalization 
            # segnp = (segnp - segnp.min()) * (1.0 - 0.0) / (segnp.max() - segnp.min())
            # segnb = nb.Nifti1Image(segnp, segnb.affine)
            # print('Normalizing image {} to [0,1]'.format(seg))
            #4.2 Resample the image to certain space using spline interpolation
            output_seg_file = os.path.join(out_HR_seg_dir, seg)   
            nb.save(segnb, output_seg_file) 
            os.system('mri_convert \"{}\" \"{}\" -vs {} {} {} -rt nearest'.format(output_seg_file, output_seg_file, self.HR_pixdim[0], self.HR_pixdim[1], self.HR_pixdim[2]))
                 
        # Padding or crop to certain size 
        print('Fine tune all volumes to trainable size')
        for seg in os.listdir(out_HR_seg_dir):
            # Image 
            seg_path = os.path.join(out_HR_seg_dir, seg)
            segnb = nb.load(seg_path)
            dim_x = segnb.shape[0]
            dim_y = segnb.shape[1]
            dim_z = segnb.shape[2]

            # Set the volume in the center, or crop from center
            x_ori = (dim_x - self.HR_dim[0]) / 2 # Floor operation 
            y_ori = (dim_y - self.HR_dim[1]) / 2
            

            os.system('fsl5.0-fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
                seg_path, seg_path, x_ori, finesize[0], y_ori, finesize[1], 0, dim_z
            ))
            print('Fine tuning image {} from {} {} {} to {}'.format(seg, dim_x, dim_y, dim_z, str(self.HR_dim)))
        
        print('Preprocessing Complete!')

