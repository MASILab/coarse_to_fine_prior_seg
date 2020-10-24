"""
The script for implement body part regression
1) Convert each volume into 2d z direction images
2) Generate a list file for all slices
3) Run model for each slice, give a score
4) Generate a txt file to save all the image names and scores

Yucheng Tang
OCT 2018

"""

import torch

import os
import numpy as np
import nibabel as nb
from PIL import Image
import matplotlib.pyplot as plt
from img_loader import img_loader
from Unet_BN import Unet_BN
import tqdm
from torch.autograd import Variable
import torch.nn.functional as F


class Bodypart_regressor(object):
### 1. ----------- COnvert original to 2d for body part regression ---------- ###
    def __init__(self, args):
        """Initialize and regressor model."""
        self.data_dir = args.data_dir
        self.txt_info_dir = args.txt_info_dir
        self.checkpoint_BPR_dir = args.checkpoint_BPR_dir

    def processing(self):
        soft_images_dir = os.path.join(self.data_dir, 'soft_images')
        image2d_dir = os.path.join(self.data_dir, 'image2d')
        #Delete the test_data directory and create a new one for current task
        os.system('rm -rf \"{}\"'.format(image2d_dir))
        os.makedirs(image2d_dir)
        indexerrorlist = []
        count = 0
        for image in os.listdir(soft_images_dir):
            if image.endswith('.nii.gz'):
                count += 1
                print('[{}] Handling {}'.format(count, image))
                 
                image_path = os.path.join(soft_images_dir, image)                
                # Convert to 2d slices for body part regressor
                imgnb = nb.load(image_path)
                imgnp = np.array(imgnb.dataobj)
                try:
                    slicenum = imgnp.shape[2]
                except IndexError:
                    print('[{}] Ignore {}'.format(count, image))
                    indexerrorlist.append(image)
                    continue
                    
                final_slice_file = os.path.join(image2d_dir, '{}_idx{}.png'.format(image,slicenum-1))
                if os.path.isfile(final_slice_file):
                    print('Skip {}'.format(image))
                    continue
                for i in range(slicenum):
                    imgnp2d = imgnp[:,:,i] # int16 numpy array
                    # Normalize the data to 0~255 space
                    try:
                        imgnp2d = (imgnp2d - imgnp2d.min()) * (255.0 - 0.0) / (imgnp2d.max() - imgnp2d.min())
                        imgnp2d = imgnp2d.astype(np.uint8)
                        image2d = Image.fromarray(imgnp2d).rotate(90)
                    except TypeError:
                        print('TypeError, cannot handle this type')
                        continue
                    output_path = os.path.join(image2d_dir, '{}_idx{}.png'.format(image, i))
                    print('[{}] Saving {}_{} to test_data '.format(count, image, i))
                    image2d.save(output_path)

        ## 2. ------------ Script for running body part regressor ---------- ###
        #1) Generate image list
        
        txt_file = os.path.join(self.txt_info_dir, 'test_image_list.txt')
        list_file = open(txt_file, 'w')
        for image in os.listdir(image2d_dir):
            list_file.write(image + '\n')
        list_file.close()
        print("Generating image list......")

        # Testing with checkpoint BPR
        fp = open(txt_file, 'r')
        sublines = fp.readlines()
        sub_infos = []
        for subline in sublines:
            sub_info = []
            sub_name = subline.split('\n')[0]
            sub_path = os.path.join(image2d_dir, sub_name)

            sub_info.append(sub_name)
            sub_info.append(sub_path)
            sub_infos.append(sub_info)
        fp.close()

        
        test_set = img_loader(sub_infos)
        self.test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=1)

        # 2. -------- Load model file
        for model in os.listdir(self.checkpoint_BPR_dir):
            model_pth = os.path.join(self.checkpoint_BPR_dir, model)

        model = Unet_BN(n_class=1) #lr = 0.0014

        cuda = torch.cuda.is_available()

        #cuda = False
        torch.manual_seed(1)
        if cuda:
            torch.cuda.manual_seed(1)
            model = model.cuda()

        model.load_state_dict(torch.load(model_pth))
        model.train()


        # load CUDA
        # 3. ----------- Testing -------------

        result_file = os.path.join(self.txt_info_dir, 'BPR_result_list.txt')
        score_list = open(result_file, 'w')

        for batch_idx, (data,img_name) in tqdm.tqdm(
                enumerate(self.test_loader), total=len(self.test_loader),
                desc='Valid epoch=150', ncols=80,
                leave=False):

            if cuda:
                data= data.cuda()
            data = Variable(data,volatile=True)

            pred = model(data)

            pred_scores = pred.cpu().data.numpy()
            for idx in range(len(img_name)):
                print(img_name[0] + ': {}'.format(pred_scores[idx][0][0][0]))
                score_list.write(str(img_name[idx]) + '\t' + str(pred_scores[idx][0][0][0]) + '\n')
        score_list.close()


