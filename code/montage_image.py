import os
import numpy as np
import nibabel as nb
from PIL import Image

data_dir = 
out_dir = 
for case in os.listdir(data_dir):
    if 'txt' not in case:
        case_dir = os.path.join(data_dir, case)
        for seq in os.listdir(case_dir):
            seq_dir = os.path.join(case_dir, seq)
            for scan in os.listdir(seq_dir):
                scan_dir = os.path.join(seq_dir, scan)
                for f in os.listdir(scan_dir):
                    if f.endswith('nii'):
                        file_path = os.path.join(scan_dir, f)
                        nii_file = nb.load(file_path)
                        nii_np = np.array(nii_file.dataobj)
                        z_range = nii_np.shape[2]
                        centre_slice_idx = z_range / 2
                        cen_slice = nii_np[:,:,centre_slice_idx]
                        cen_slice_file = os.path.join(out_dir, f + '.png')
                        cen_slice_image = Image.fromarray(cen_slice).convert('RGB')
                        cen_slice_image.save(cen_slice_file)
                        print('saving {}'.format(f))


count = 0 
label_dir = 
output_dir = 
for label in os.listdir(label_dir):
    count += 1
    label_file = os.path.join(label_dir, label)
    labelnb = nb.load(label_file)
    labelnp = np.array(labelnb.dataobj)

    image_file = label_file.replace('_seg','')
    image_file = image_file.replace('/labels/', '/soft_images/')
    imgnb = nb.load(image_file)


    lablenb_new = nb.Nifti1Image(labelnp, imgnb.affine)
    label_newfile = os.path.join(output_dir, label)
    nb.save(lablenb_new, label_newfile)
    print('[{}] Converting {} to space'.format(count, label))