import nibabel as nb
import numpy as np
import os
from PIL import Image

label_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/12sigma'
output_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/eso_nifti'

for label in os.listdir(label_dir):
    label_file = os.path.join(label_dir, label)
    label_nb = nb.load(label_file)
    label_np = np.array(label_nb.dataobj)
    for organ_idx in range(1, 9):
        idx = np.where(label_np == organ_idx)
        label_np[idx] = 0
    for organ_idx in range(10, 15):
        idx = np.where(label_np == organ_idx)
        label_np[idx] = 0
    lablenb_new = nb.Nifti1Image(label_np, label_nb.affine)
    label_newfile = os.path.join(output_dir, label)
    nb.save(lablenb_new, label_newfile)



image_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/soft_images'
output_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/eso_nifti'
for image in os.listdir(image_dir):
    imgnb = nb.load(os.path.join(image_dir, image))
    label_file = os.path.join(output_dir, image)
    labelnb = nb.load(label_file)
    labelnp = np.array(labelnb.dataobj)


    lablenb_new = nb.Nifti1Image(labelnp, imgnb.affine)
    label_newfile = os.path.join(output_dir, image)
    nb.save(lablenb_new, label_newfile)



image_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/soft_images'
output_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/2d_slices_eso'
label_dir = '/share5/tangy5/ImageVU_abdomen/pipeline/segout/seg/eso_nifti'

for image in os.listdir(image_dir):
    imgnb = nb.load(os.path.join(image_dir, image))
    imgnp = np.array(imgnb.dataobj)

    label_file = os.path.join(label_dir, image)
    labelnb = nb.load(label_file)
    labelnp = np.array(labelnb.dataobj)

    for slice_idx in range(imgnp.shape[2]):
        imgnp_slice = imgnp[:,:,slice_idx]
        labelnp_slice1 = labelnp[:,:,slice_idx]
        labelnp_slice2 = np.zeros((labelnp_slice1.shape[0], labelnp_slice1.shape[1]))

        imgnp_slice = (imgnp_slice - imgnp_slice.min()) * (255.0 - 0.0) / (imgnp_slice.max() - imgnp_slice.min())
        imgnp_slice = imgnp_slice.astype(np.uint8)



        labelnp_slice1 = (labelnp_slice1 - labelnp_slice1.min()) * (255.0 - 0.0) / (labelnp_slice1.max() - labelnp_slice1.min())
        labelnp_slice1 = labelnp_slice1.astype(np.uint8)
        labelnp_slice2 = labelnp_slice2.astype(np.uint8)

        labelnp_slice = np.zeros((labelnp_slice1.shape[0], labelnp_slice1.shape[1], 3))
        labelnp_slice[:,:,0] = labelnp_slice1
        labelnp_slice[:,:,1] = labelnp_slice2
        labelnp_slice[:,:,2] = labelnp_slice2
        labelnp_slice = labelnp_slice.astype(np.uint8)


        background = Image.fromarray(imgnp_slice).rotate(90).convert("RGB")

        overlay = Image.fromarray(labelnp_slice).rotate(90)

        new_slice = Image.blend(background, overlay, 0.5)

        case_dir = os.path.join(output_dir, image)
        if not os.path.isdir(case_dir):
            os.makedirs(case_dir)
        new_image_path = os.path.join(case_dir, '{}_slice_{}.png'.format(image, slice_idx))
        new_slice.save(new_image_path, "PNG")
        print("Saving {}_{} ".format(image, slice_idx))



