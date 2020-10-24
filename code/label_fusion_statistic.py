"""
Majority vote is implemented here, and make summary of all number of patches results

Yucheng Tang
May 2019

"""

import os
import numpy as np
import nibabel as nb


class Label_fusion(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.checkpoint_randomPatch = args.checkpoint_randomPatch_dir
        self.segout_dir = args.segout_dir
        self.txt_info_dir = args.txt_info_dir
        self.randPatch_dir = args.randPatch_dir
        # self.targetOrgan = args.targetOrgan
    
    def mkdir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def processing(self):
        bpr_crop_file = os.path.join(self.txt_info_dir, 'bpr_crop_index.txt')
        with open(bpr_crop_file) as f:
            content2 = f.readlines()
            content2 = [x.strip() for x in content2]
            bpr_dict = {}
            for bpr_item in content2:
                item_name = bpr_item.split(' ')[0]
                bpr_z_start = int(bpr_item.split(' ')[1])
                bpr_z_end = int(bpr_item.split(' ')[2])
                original_z_dim = int(bpr_item.split(' ')[3])
                bpr_dict[item_name] = []
                bpr_dict[item_name].append(bpr_z_start)
                bpr_dict[item_name].append(bpr_z_end)
                bpr_dict[item_name].append(original_z_dim)
        
        index2organ = {1 : 'spleen', 2 : 'rk',  3 : 'lk', 4 : 'gall',  5 : 'eso',  6 : 'liver', \
            7 : 'stomach', 8 : 'aorta', 9 : 'IVC', 10 : 'PSV', 11 : 'pancreas', 12 : 'rad', 13 : 'lad'}

        organ2index = {'spleen' : 1, 'rk' : 2, 'lk' : 3, 'gall' : 4, 'eso' : 5, 'liver' : 6, \
            'stomach' : 7, 'aorta' : 8, 'IVC' : 9, 'PSV' : 10, 'pancreas' : 11, 'rad' : 12, 'lad' : 13}

        random_patches_num = 50
        patch_size = [128,128,48]
        image_dict = {}

        for i in range(1, 14):
            self.targetOrgan = i
            segout_dir = os.path.join(self.randPatch_dir, 'patch_seg_{}'.format(self.targetOrgan))
            # if not os.path.isdir(segout_dir):
            #     image_dict[image_name][self.targetOrgan] = []
    
            # 2. Sort segs with name
            for seg in os.listdir(segout_dir):
                seg_file = os.path.join(segout_dir, seg)

                # if i == 13:
                    # image_name = seg.split('.gz')[0]
                # else:
                image_name = seg.split('.nii.gz_')[0]
                image_name = image_name + '.nii.gz'
                if image_name not in image_dict:
                    image_dict[image_name] = {}
                if self.targetOrgan not in image_dict[image_name]:
                    image_dict[image_name][self.targetOrgan] = []
                image_dict[image_name][self.targetOrgan].append(seg_file)
            #print(image_dict)

        count = 0
        for image in image_dict:
            count += 1

            image_dir = os.path.join(self.cropped_dir, 'soft_images')

            output_dir = os.path.join(self.randPatch_dir, 'fusion_seg')
            self.mkdir(output_dir)

            output_path = os.path.join(output_dir, image)
            if os.path.isfile(output_path):
                print('[{}] File exsits, skiping {}'.format(count, image))
                continue
            image_name = image

            image_file = os.path.join(image_dir, image_name)
            imagenb = nb.load(image_file)
            imagenp = np.array(imagenb.dataobj)

            image_x = imagenp.shape[0]
            image_y = imagenp.shape[1]
            image_z = imagenp.shape[2]


            label_fusion_np = np.zeros((14, imagenp.shape[0], imagenp.shape[1], imagenp.shape[2]))

            for i in range(1, 14):
                self.targetOrgan = i
            # 3. load random patches crop coor txt file to a dictionary
                patch_coor_txt_file = os.path.join(self.txt_info_dir, 'patches_128_50_coor_{}.txt'.format(self.targetOrgan))
    
                with open(patch_coor_txt_file) as f:
                    content = f.readlines()
                content = [x.strip() for x in content]
    
                subject2patchCoor = {}
                for item in content:
                    print(item)
                    image_name = item.split(' ')[0]
                    patch_number = int(item.split(' ')[1].split(':')[1])
                    patch_coor_parts = item.split(' ')[2:]
                    patch_coor = []
                    patch_coor.append(patch_number)
                    patch_coor.append(int(patch_coor_parts[0]))
                    patch_coor.append(int(patch_coor_parts[1]))
                    patch_coor.append(int(patch_coor_parts[2]))
                    if image_name not in subject2patchCoor:
                        subject2patchCoor[image_name] = []
                    subject2patchCoor[image_name].append(patch_coor)
                #print(subject2patchCoor)
    
                if self.targetOrgan not in image_dict[image]:
                    print('[{}] image has no {}'.format(count, self.targetOrgan))
                    continue
                for idx, seg_file in enumerate(image_dict[image][self.targetOrgan]):
                    if idx < random_patches_num:
                        if i == 13: 
                            seg_patch_idx = int(seg_file.split('.nii.gz_')[1].split('_')[1].split('.nii.gz')[0])
                            segname = seg_file.split('/')[-1].split('.nii.gz_')[0] + '.nii.gz'
                        else:
                            seg_patch_idx = int(seg_file.split('.nii.gz_')[1].split('_')[1].split('.nii.gz')[0])
                        #seg_patch_idx = int(seg_file.split('/')[-1].split('_')[-1].split('.nii.gz')[0])
                            segname = seg_file.split('/')[-1].split('.nii.gz_')[0] + '.nii.gz'
                        # segname = segname.replace('.gz', '.nii.gz')
                        patchNumber2coor_lists = subject2patchCoor[segname]
                        #find corresponding pathch coor
                        for patch_coor in patchNumber2coor_lists:
                            if patch_coor[0] == seg_patch_idx:
                                seg_coor = patch_coor[1:]
                        
                        segnb = nb.load(seg_file)
                        segnp = np.array(segnb.dataobj)
    
                        # indices1 = np.where(segnp == 1)
                        # segnp[indices1] = 2
                        segnp = segnp.astype(np.int8)
                        indices0 = np.where(segnp == 0)
                        segnp[indices0] = -1
    
    
    
                        x_ori = int(seg_coor[0]-patch_size[0]/2)
                        x_end = int(x_ori + patch_size[0])
                        y_ori = int(seg_coor[1]-patch_size[1]/2)
                        y_end = int(y_ori + patch_size[1])
                        z_ori = int(seg_coor[2]-patch_size[2]/2)
                        z_end = int(z_ori + patch_size[2])
    
                        seg_x_ori = 0
                        seg_y_ori = 0
                        seg_z_ori = 0
                        seg_x_end = segnp.shape[0]
                        seg_y_end = segnp.shape[1]
                        seg_z_end = segnp.shape[2]
    
                        if x_ori < 0:
                            x_distance = 0 - x_ori
                            x_ori = 0
                            seg_x_ori = x_distance
                        if x_end > 512:
                            x_distance = x_end - 512
                            x_end = 512
                            seg_x_end = segnp.shape[0] - x_distance
                        if y_ori < 0:
                            y_distance = 0 - y_ori
                            y_ori = 0
                            seg_y_ori = y_distance
                        if y_end > 512:
                            y_distance = y_end - 512
                            y_end = 512
                            seg_y_end = segnp.shape[1] - y_distance
                        if z_ori < 0:
                            z_distance = 0 - z_ori
                            z_ori = 0
                            seg_z_ori = z_distance
                        if z_end > imagenp.shape[2]:
                            z_distance = z_end - imagenp.shape[2]
                            z_end = imagenp.shape[2]
                            seg_z_end = segnp.shape[2] - z_distance
    
                        # print(seg_z_ori)
                        # print(seg_z_end)
                        # print(imagenp.shape)
                        # print(segnp[seg_x_ori:seg_x_end,\
                        #     seg_y_ori:seg_y_end,seg_z_ori:seg_z_end].shape)
                        # print(label_fusion_np[int(self.targetOrgan)][x_ori:x_end, y_ori:y_end,z_ori:z_end].shape)
                        # print(x_ori)
                        # print(x_end)
                        # print(y_ori)
                        # print(y_end)
                        # print(z_ori)
                        # print(z_end)
                        label_fusion_np[int(self.targetOrgan)][x_ori:x_end, y_ori:y_end,z_ori:z_end] += segnp[seg_x_ori:seg_x_end,\
                            seg_y_ori:seg_y_end,seg_z_ori:seg_z_end]
                   
                        print('[{}---{}---{}] segmentation patched back {}'.format(count, self.targetOrgan, seg_patch_idx, image))
                # single Majority vote here

            final_multi_organ_np = label_fusion_np.argmax(0)
            # print(final_multi_organ_np.min())

            final_multi_organ_np = final_multi_organ_np.astype(np.float32)

            # Save final numpy
            final_labelnb = nb.Nifti1Image(final_multi_organ_np, imagenb.affine)
            output_path = os.path.join(output_dir, image)
            nb.save(final_labelnb, output_path)
            print('[{}] joint label fusion complete {}'.format(count, image))
           
            # z_ori = 0-bpr_dict[segname][0]
            # original_dim_z = bpr_dict[segname][2]
            # os.system('fslroi \"{}\" \"{}\" {} {} {} {} {} {}'.format(
            #             output_path, output_path, 0, 512, 0, 512, z_ori, original_dim_z
            #         ))

            # print('[{}] back to full volume {} from {} {} {} to {} {} {}'.format(count, image, final_labelnb.shape[0],\
            #     final_labelnb.shape[1], final_labelnb.shape[2], 512, 512, original_dim_z))





        # patch number 1 to 50 , Majority vote here
            
        # final_label = np.zeros((50, imagenp.shape[0], imagenp.shape[1], imagenp.shape[2]))
        # count_voxel = 0
        # for i in range(imagenp.shape[0]):
        #     print(i)
        #     for j in range(imagenp.shape[1]):
        #         for k in range(imagenp.shape[2]):

        #             for number in range(1, 51):

        #                 # label_fusion_np_selected = label_fusion_np[:number,:,:,:]
        #                 voxelVec = label_fusion_np[:number,i,j,k]

        #                 unique, counts = np.unique(voxelVec, return_counts=True)
        #                 majority_array = [0,0,0]
        #                 for index, ele in enumerate(unique):
        #                     if ele == 0:
        #                         majority_array[0] = counts[index]
        #                     elif ele == 1:
        #                         majority_array[1] = counts[index]
        #                     else:
        #                         majority_array[2] = counts[index]
        #                 # Only if there are more than 1 voters vote yes and yes > no, that voxel can
        #                 # be marked yes
        #                 voxVal = 0
        #                 if majority_array[2] > 1 and majority_array[2] > majority_array[1]:
        #                     voxVal = 1
        #                 final_label[number - 1][i,j,k] = voxVal
        #                 voxVal = 0

        # # Save 50 number 
        # for vol_idx in range(1,51):
        #     final_labelnb = nb.Nifti1Image(final_label[vol_idx-1], imagenb.affine)
        #     image_index_name = image.replace('.nii.gz', '_{}.nii.gz'.format(vol_idx))
        #     output_path = os.path.join(output_dir, image_index_name)
        #     nb.save(final_labelnb, output_path)
        #     print('[{}] joint label fusion complete {}'.format(count, image_index_name))


























