"""
A pipeline software for bodypart regression and preprocessing

Yucheng Tang
OCT 2018

"""
from __future__ import print_function
import os
import numpy as np
#import caffe
import nibabel as nb
from nibabel.processing import resample_from_to 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from pre_bodypart_regression import Pre_bodypart
from bodypart_regression import Bodypart_regressor
from post_bodypart import post_bodypart_regression
from resample_nifti import Resample_nifti

from resample_nifti_HR import Resample_nifti_HR
# from preSegs import PreSegs

from test import Testor
#  from test_context import Testor_context
import torch
import torch._utils

# from cropHRfromGT import PreSegs

from post_processing import Recoveror

from random_patch import Random_sampler
from test_randomPatch import Testor_randomPatch
from label_fusion_statistic import Label_fusion

### ---------- Parameters ---------- ###
import argparse
parser = argparse.ArgumentParser(description='Scripts pipeline for abodomen CT preprocessing')
# All working dirs
parser.add_argument('--data_dir',  default='../INPUTS', 
                    help='root path for first volume which contains the middle index ')

             
parser.add_argument('--checkpoint_dir', default= '../checkpoint',
                    help='Folder that store all preprocessed niftis for training')  
parser.add_argument('--checkpoint_BPR_dir', default= '../checkpoint_BPR',
                    help='Folder that store all preprocessed niftis for training')  
parser.add_argument('--checkpoint_randomPatch_dir', default= '../checkpoint_randomPatch',
                    help='Folder that store random patch models')  

parser.add_argument('--cropped_dir', default= '../INPUTS/cropped',
                    help='Folder that stored all body part regressed volume')

parser.add_argument('--output_dir', default= '../OUTPUTS',
                    help='Folder that store all preprocessed niftis for training')  


parser.add_argument('--segout_dir', default= '../OUTPUTS/segout',
                    help='Folder that store all preprocessed niftis for training')  
parser.add_argument('--txt_info_dir', default= '../txt_info',
                    help='Folder that store all preprossing txt files')  
parser.add_argument('--dim', default='168 168 64',
                    help='The output dimension') 
parser.add_argument('--pixdim', default='2 2 6',
                    help='spacing if required consistent pixdimension')

parser.add_argument('--HR_dim', default='512 512 0',
                    help='The output dimension') 
parser.add_argument('--HR_pixdim', default='1 1 2',
                    help='spacing if required consistent pixdimension')

parser.add_argument('--score_interval', default='-4 5', 
                    help='body part regression score select')                                                            


parser.add_argument('--randPatch_dir', default='../randpatch', 
                    help='body part regression score select')  
# parser.add_argument('--targetOrgan', default='8', 
#                     help='1:spleen,2:rk,3:lk,4:gall,5:eso,6:liver,7:stomach,8:aorta,9:IVC,10:PSV,11:pancreas,12:rad,13:lad')    


args = parser.parse_args()

if not os.path.isdir(args.cropped_dir):
    os.makedirs(args.cropped_dir)
if not os.path.isdir(os.path.join(args.cropped_dir, 'images')):
    os.makedirs(os.path.join(args.cropped_dir, 'images'))
if not os.path.isdir(os.path.join(args.cropped_dir, 'soft_images')):
    os.makedirs(os.path.join(args.cropped_dir, 'soft_images'))    
if not os.path.isdir(args.randPatch_dir):
    os.makedirs(args.randPatch_dir)  

Pre_bodypart = Pre_bodypart(args)
Pre_bodypart.processing()
Bodypart_regressor = Bodypart_regressor(args)
Bodypart_regressor.processing()
post_bodypart_regression = post_bodypart_regression(args)
post_bodypart_regression.processing()

Resample_nifti = Resample_nifti(args)
Resample_nifti.processing()
Testor = Testor(args)
Testor.processing()
context_test = Testor_context(args)
context_test.processing()
Recoveror = Recoveror(args)
Recoveror.processing()

Random_sampler = Random_sampler(args)

Random_sampler.processing()

Testor_randomPatch = Testor_randomPatch(args)
Testor_randomPatch.processing()

Label_fusion = Label_fusion(args)
Label_fusion.processing()



