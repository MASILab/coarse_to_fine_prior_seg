#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:58:47 2019

@author: leeh43
"""

import os 
import numpy as np
import nibabel as nib

main_dir = '/nfs/masi/leeh43'
img_dir = '100_random_ImageVU_B'
data_dir = os.path.join(main_dir, img_dir)
datalist = []
for img in os.listdir(data_dir):
    image = os.path.join(data_dir, img)
    image_nii = nib.load(image)
    image_header = image_nii.get_zoom()