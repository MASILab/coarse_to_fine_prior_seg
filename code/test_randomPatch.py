import torch
import os
from glob import glob
import tqdm
from imgloader_randomPatch_3D import pytorch_loader
from Unet3D import UNet3D
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import nibabel as nib

class Testor_randomPatch(object):
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
        for i in range(1, 14):
            self.targetOrgan = str(i)
            index2organ = {1 : 'spleen', 2 : 'rk',  3 : 'lk', 4 : 'gall',  5 : 'eso',  6 : 'liver', \
                7 : 'stomach', 8 : 'aorta', 9 : 'IVC', 10 : 'PSV', 11 : 'pancreas', 12 : 'rad', 13 : 'lad'}

            organ2index = {'spleen' : 1, 'rk' : 2, 'lk' : 3, 'gall' : 4, 'eso' : 5, 'liver' : 6, \
                'stomach' : 7, 'aorta' : 8, 'IVC' : 9, 'PSV' : 10, 'pancreas' : 11, 'rad' : 12, 'lad' : 13}
            # 1.-------- Set testing loader --------------
            all_test_img_subs = []
            all_test_img_files = []
            all_test_seg_subs = []
            all_test_seg_files = []
            all_test_img_resource = []
            all_test_seg_resource = []

            test_img_dir = os.path.join(self.randPatch_dir, 'softimg_patches_{}'.format(self.targetOrgan))
            test_seg_dir = os.path.join(self.randPatch_dir, 'seg_patches_{}'.format(self.targetOrgan))

            image_list = []
            image_files = glob(os.path.join(test_img_dir,"*.nii.gz"))
            image_files.sort()
            for name in image_files:
                image_list.append(os.path.basename(name)[:-7]) 
            test_img_subs, test_img_files = image_list, image_files

            seg_list = []
            seg_files = glob(os.path.join(test_seg_dir,"*.nii.gz"))
            seg_files.sort()
            for name in seg_files:
                seg_list.append(os.path.basename(name)[:-7])
            test_seg_subs, test_seg_files = seg_list, seg_files

                # Must keep order of each list above
            for item in test_img_subs:
                all_test_img_subs.append(item)
            for item in test_img_files:
                all_test_img_files.append(item)

            for item in test_seg_subs:
                all_test_seg_subs.append(item)
            for item in test_seg_files:
                all_test_seg_files.append(item)

            test_dict = {}
            test_dict['img_subs'] = all_test_img_subs
            test_dict['img_files'] = all_test_img_files
            test_dict['pred_subs'] = all_test_seg_subs
            test_dict['pred_files'] = all_test_seg_files


            test_set = pytorch_loader(test_dict, num_labels=2)
            self.test_loader = torch.utils.data.DataLoader(test_set,batch_size=2,shuffle=False,num_workers=1)

            # 2. -------- Load model file
            organ_name = index2organ[int(self.targetOrgan)]
            for model in os.listdir(self.checkpoint_randomPatch):
                # if organ_name in model:
                model_pth = os.path.join(self.checkpoint_randomPatch, model)



            model = UNet3D(in_channel=2, n_classes=2)

            cuda = torch.cuda.is_available()

            #cuda = False
            torch.manual_seed(1)
            if cuda:
                torch.cuda.manual_seed(1)
                model = model.cuda()

            model.load_state_dict(torch.load(model_pth))
            model.train()

            with torch.no_grad():
            # load CUDA
            # 3. ----------- Testing -------------
                for batch_idx, (data,target,sub_name) in tqdm.tqdm(
                        # enumerate(self.test_loader), total=len(self.test_loader),
                        enumerate(self.test_loader), total=len(self.test_loader),
                        desc='Valid', ncols=80,
                        leave=False):
                    print(sub_name)
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data,volatile=True), Variable(target,volatile=True)
                    pred = model(data)

                    # Save all labels to one NIFTI file
                    lbl_pred = pred.data.max(1)[1].cpu().numpy()[:,:, :].astype('uint8')
                    lbl_pred = np.transpose(lbl_pred, (0, 2, 3, 1))

                    # pred_prob = pred.data.max(1)[0].cpu().numpy()[:,:,:]
                    # pred_prob = np.transpose(pred_prob, (0, 2, 3, 1))

                    batch_num = lbl_pred.shape[0]
                    for si in range(batch_num):
                        curr_sub_name = sub_name[si]

                        out_img_dir = os.path.join(self.randPatch_dir, 'patch_seg_{}'.format(self.targetOrgan))
                        self.mkdir(out_img_dir)

                        # out_prob_dir = os.path.join(self.randPatch_dir, 'prob_seg_{}'.format(self.targetOrgan))
                        # self.mkdir(out_prob_dir)     

                        out_nii_file = os.path.join(out_img_dir,('%s.nii.gz'%(curr_sub_name)))
                        seg_img = nib.Nifti1Image(lbl_pred[si], affine=np.eye(4))
                        nib.save(seg_img, out_nii_file)

                        # out_prob_nii_file = os.path.join(out_prob_dir,('%s.nii.gz'%(curr_sub_name)))
                        # seg_prob = nib.Nifti1Image(pred_prob[si], affine=np.eye(4))
                        # nib.save(seg_prob, out_prob_nii_file)

                        print('Saving all labels to one NIFTI')

                # Allign predicted segmentation with image affine
                # print('Aligning predicted seg with image affine')
                # out_seg_dir = os.path.join(self.randPatch_dir, 'patch_seg')
                # count = 0
                # for seg in os.listdir(out_seg_dir):
                #     count += 1
                #     seg_name = seg.replace('_seg', '')

                #     seg_file = os.path.join(out_seg_dir, seg)
                #     soft_image_file = os.path.join(test_img_dir, seg_name)

                #     segnb = nib.load(seg_file)
                #     segnp = np.array(segnb.dataobj)
                #     imgnb = nib.load(soft_image_file)

                #     seg_img = nib.Nifti1Image(segnp, affine=imgnb.affine)
                #     nib.save(seg_img, seg_file)
                #     print('[{}] Saved aligned segmentation map with image {}'.format(count, seg))
