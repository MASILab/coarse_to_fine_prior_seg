import torch
import os
from glob import glob
import tqdm
from imgloader_InCyte_MRI_3D import pytorch_loader
from Unet3D import UNet3D
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import nibabel as nib

class Testor_context(object):
    def __init__(self, args):
        """Resample the nifti files ."""
        self.cropped_dir = args.cropped_dir
        self.output_dir = args.output_dir
        self.checkpoint = args.checkpoint_dir
        self.segout_dir = args.segout_dir
    
    def mkdir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
    
    def processing(self):

        # 1.-------- Set testing loader --------------
        all_test_img_subs = []
        all_test_img_files = []
        all_test_seg_subs = []
        all_test_seg_files = []
        all_test_img_resource = []
        all_test_seg_resource = []

        test_img_dir = os.path.join(self.output_dir, 'soft_images')
        context_img_dir = os.path.join(self.segout_dir, 'seg')

        image_list = []
        image_files = glob(os.path.join(test_img_dir,"*.nii.gz"))
        image_files.sort()

        for name in image_files:
            image_list.append(os.path.basename(name)[:-7])

        
        test_img_subs, test_img_files = image_list, image_files


            # Must keep order of each list above
        for item in test_img_subs:
            all_test_img_subs.append(item)
        for item in test_img_files:
            all_test_img_files.append(item)

        test_dict = {}
        test_dict['img_subs'] = all_test_img_subs
        test_dict['img_files'] = all_test_img_files
        
        test_set = pytorch_loader(test_dict)
        self.test_loader = torch.utils.data.DataLoader(test_set,batch_size=1,shuffle=False,num_workers=1)

        # 2. -------- Load model file
        for model in os.listdir(self.checkpoint):
            model_pth = os.path.join(self.checkpoint, model)



        model = UNet3D(in_channel=14, n_classes=14)

        cuda = torch.cuda.is_available()

        #cuda = False
        torch.manual_seed(1)
        if cuda:
            torch.cuda.manual_seed(1)
            model = model.cuda()
            model = torch.nn.DataParallel(model)

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
                iter_context = os.path.join(context_img_dir, str(sub_name[0]) + '.nii.gz')
                context_nib = nib.load(iter_context)
                context_np = context_nib.get_data()
                context_np = np.transpose(context_np, (2, 0, 1))
                ori_data = data.data.cpu().numpy()
                # print(ori_data.shape)
                for i in range(1,14):
                    pred_one = context_np == i
                    pred_one[pred_one!=0] = 1
                    pred_one = pred_one.astype('float32')
                    ori_data[:, i, :, :, :] = pred_one[:, :, :]
                new_data = torch.Tensor(ori_data)
                new_data, target = new_data.cuda(), target.cuda()
                new_data, target = Variable(new_data,volatile=True), Variable(target,volatile=True)
                pred = model(new_data)

                # Save all labels to one NIFTI file
                lbl_pred = pred.data.max(1)[1].cpu().numpy()[:,:, :].astype('uint8')
                lbl_pred = np.transpose(lbl_pred, (0, 2, 3, 1))
                batch_num = lbl_pred.shape[0]
                for si in range(batch_num):
                    curr_sub_name = sub_name[si]
                    out_img_dir = os.path.join(self.segout_dir, 'seg_2')
                    self.mkdir(out_img_dir)
                    out_nii_file = os.path.join(out_img_dir,('%s.nii.gz'%(curr_sub_name)))
                    seg_img = nib.Nifti1Image(lbl_pred[si], affine=np.eye(4))
                    nib.save(seg_img, out_nii_file)
                    print('Saving all labels to one NIFTI')

            # Allign predicted segmentation with image affine
            print('Aligning predicted seg with image affine')
            out_seg_dir = os.path.join(self.segout_dir, 'seg_2')
            count = 0
            for seg in os.listdir(out_seg_dir):
                count += 1
                seg_name = seg.replace('_seg', '')

                seg_file = os.path.join(out_seg_dir, seg)
                soft_image_file = os.path.join(test_img_dir, seg_name)

                segnb = nib.load(seg_file)
                segnp = np.array(segnb.dataobj)
                imgnb = nib.load(soft_image_file)

                seg_img = nib.Nifti1Image(segnp, affine=imgnb.affine)
                nib.save(seg_img, seg_file)
                print('[{}] Saved aligned segmentation map with image {}'.format(count, seg))



# # check nifti avaliable
# data_dir = ''
# count = 0
# for image in os.listdir(data_dir):
#     count += 1
#     image_path = os.path.join(data_dir, image)
#     imagenb = nb.load(image_path)
#     print('[{}] image name --- {}'.format(count, image))
#     imagenp = np.array(imagenb.dataobj)
#     print('[{}] Valid image  dim [{}] [{}] [{}]'.format(count, imagenp.shape[0], \
#         imagenp.shape[1], imagenp.shape[2]))
