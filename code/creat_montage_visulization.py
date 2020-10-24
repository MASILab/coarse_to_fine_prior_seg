import numpy as np
import nibabel as nb
import os
from PIL import Image




colorpick = [[255,30,30],[255,245,71],[112,255,99],[9,150,37],[30,178,252],[132,0,188],\
    [255,81,255],[158,191,9],[255,154,2],[102,255,165],[0,242,209],[255,0,80],[40,0,242]]
#Spleen: red,right kid: yellow, left kid green, gall:sky blue, eso:blue,liver:lg blue
#sto:pink,aorta: purple,IVC, potal vein: orange, pancreas: favor, adrenal gland
organ_list = ['spleen', 'right_kigney', 'left_kidney', 'gallbladder', 'esophagus', 'liver',\
                'stomach', 'aorta', 'IVC', 'veins', 'pancreas', 'adrenal_gland', 'all_organs']

image_dir = ''
seg_dir = ''
overlay_dir = ''

# 1 --- Align segmentation to same NIFT header with image
count = 0
for seg in os.listdir(seg_dir):
    seg_file = os.path.join(seg_dir, seg)
    segnb = nb.load(seg_file)
    segnp = np.array(segnb.dataobj)

    image_path = os.path.join(image_dir, seg)
    imgnb = nb.load(image_path)

    segnb_new = nb.Nifti1Image(segnp, imgnb.affine)
    seg_newfile = os.path.join(seg_dir, seg)
    nb.save(segnb_new, seg_newfile)
    count += 1
    print('[{}] --- Aligning segmentation affine -- {} '.format(count, seg))



#2--Compute organ volume
result = {}
count = 0
for seg in os.listdir(seg_dir):
    seg_file = os.path.join(seg_dir, seg)
    segnb = nb.load(seg_file)
    pixdim_x = segnb.header['pixdim'][1]
    pixdim_y = segnb.header['pixdim'][2]
    pixdim_z = segnb.header['pixdim'][3]
    count += 1
    prod_pixdim = pixdim_x * pixdim_y * pixdim_z * 24

    segnp = np.array(segnb.dataobj)
    cur_result = []
    for organ_idx in range(1,13):
        pix_count = 0
        for i in range(segnp.shape[0]):
            for j in range(segnp.shape[1]):
                for k in range(segnp.shape[2]):
                    if segnp[i,j,k] == organ_idx:
                        pix_count += 1

        volume = pix_count * prod_pixdim / 1000
        cur_result.append(volume)
    print('[{}] Anatomy volume computed'.format(count))
    result[seg] = cur_result

# 3 --- Overlay 2d images and save to folder
contrast_dir = ''
seg_contrast_dir = ''

for img in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img)
    imgnb = nb.load(img_path)
    imgnp = np.array(imgnb.dataobj)
    idx = np.where(imgnp < -175)
    imgnp[idx[0], idx[1], idx[2]] = -175 # set minmum to -175
    idx = np.where(imgnp > 275)
    imgnp[idx[0], idx[1], idx[2]] = 275 # set maximum to 275
    imgnp = (imgnp - imgnp.min()) * (1.0 - 0.0) / (imgnp.max() - imgnp.min())

    # save an copy of soft tissue window
    soft_image_path = img_path.replace('.nii.gz', '_soft.nii.gz')
    soft_nb = nb.Nifti1Image(imgnp, imgnb.affine)        
    nb.save(soft_nb, soft_image_path) 


image_dir = ''
seg_dir = ''
overlay_dir = ''


count = 0

# for con_dir in os.listdir(contrast_dir):
#     image_dir = os.path.join(contrast_dir, con_dir)
#     seg_dir = os.path.join(seg_contrast_dir, con_dir)

for img in os.listdir(image_dir):
    count += 1

    #Check exsit
    # overlay_allorgan_dir = os.path.join(overlay_dir, img, 'all_organs')
    # last_allorgan_slice = os.path.join(overlay_allorgan_dir, img + '_64.png')
    # if os.path.isdir(overlay_allorgan_dir):
    #     print('[{}] Exists, Skiping {}'.format(count, img))
    #     continue
    overlay_allorgan_file = os.path.join(overlay_dir, img, 'all_organs', img + '_63.png')
    # if os.path.isfile(overlay_allorgan_file):
    #     print('[{}] Exists, skiping {}'.format(count, img))
    #     continue
    image_path = os.path.join(image_dir, img)
    seg_file = os.path.join(seg_dir, img)
    if os.path.isfile(image_path) and os.path.isfile(seg_file):
        print('[{}] Processing {}'.format(count, img))
        imgnb = nb.load(image_path)
        imgnp = np.array(imgnb.dataobj)

        segnb = nb.load(seg_file)
        segnp = np.array(segnb.dataobj)

        z_range = imgnp.shape[2]
        
        output_case_dir = os.path.join(overlay_dir, img)
        if not os.path.isdir(output_case_dir):
            os.makedirs(output_case_dir)
        for organ in organ_list:
            output_case_organ_dir = os.path.join(output_case_dir, organ)
            if not os.path.isdir(output_case_organ_dir):
                os.makedirs(output_case_organ_dir)

        for organ_idx in range(1,14):
            current_organ = organ_list[organ_idx-1]
            #Save all organ multi-labels into all_organs folder
            if current_organ == 'all_organs':
                for i in range(z_range):
                    slice2dnp = imgnp[:,:,i] * 255
                    slice2d = Image.fromarray(slice2dnp.astype(np.uint8)).rotate(90)
                    slice2d = slice2d.convert('RGB')

                    sliceseg2d = segnp[:,:,i]
                    sliceseg2d_organs = np.zeros((12,segnp.shape[0], segnp.shape[1]))
               
                    overlayslice = np.zeros((segnp.shape[0], segnp.shape[1], 3))
                        
                    overlayslice[:,:,0] = sliceseg2d
                    overlayslice[:,:,1] = sliceseg2d
                    overlayslice[:,:,2] = sliceseg2d            
                    for organ in range(1,13):
                        indices1 = np.where(overlayslice[:,:,0] == organ)
                        overlayslice[:,:,0][indices1] = colorpick[organ-1][0]
                        indices2 = np.where(overlayslice[:,:,1] == organ)
                        overlayslice[:,:,1][indices2] = colorpick[organ-1][1]
                        indices3 = np.where(overlayslice[:,:,2] == organ)
                        overlayslice[:,:,2][indices3] = colorpick[organ-1][2]
                                    
                    overlayslice_image = Image.fromarray(overlayslice.astype(np.uint8)).rotate(90)                
                    overlay = Image.blend(slice2d, overlayslice_image, 0.4)
                    overlay_file = os.path.join(output_case_dir, current_organ, '{}_{}.png'.format(img, i))
                    overlay.save(overlay_file)
                    print('[{}] -- {} processed'.format(count, current_organ))


        # Save single lable to single organ folders
            # for i in range(z_range):
            #     slice2dnp = imgnp[:,:,i] * 255
            #     slice2d = Image.fromarray(slice2dnp.astype(np.uint8)).rotate(90)
            #     slice2d = slice2d.convert('RGB')

            #     sliceseg2d = segnp[:,:,i]

            #     sliceseg2d_organs = np.zeros((12,segnp.shape[0], segnp.shape[1]))

            #     # for organ in range(1,13):                
            #     #     indices = np.where(sliceseg2d == organ)
            #     #     sliceseg2d_organs[organ-1,:,:][indices] = 1
                
            #     overlayslice = np.zeros((segnp.shape[0], segnp.shape[1], 3))
                    
            #     overlayslice[:,:,0] = sliceseg2d
            #     overlayslice[:,:,1] = sliceseg2d
            #     overlayslice[:,:,2] = sliceseg2d

            #     indices1 = np.where(overlayslice[:,:,0] == organ_idx)
            #     overlayslice[:,:,0][indices1] = colorpick[organ_idx-1][0]
            #     indices2 = np.where(overlayslice[:,:,1] == organ_idx)
            #     overlayslice[:,:,1][indices2] = colorpick[organ_idx-1][1]
            #     indices3 = np.where(overlayslice[:,:,2] == organ_idx)
            #     overlayslice[:,:,2][indices3] = colorpick[organ_idx-1][2]

            #     overlayslice_image = Image.fromarray(overlayslice.astype(np.uint8)).rotate(90)
                    
            #     overlay = Image.blend(slice2d, overlayslice_image, 0.4)

            #     overlay_file = os.path.join(output_case_dir, current_organ, '{}_{}.png'.format(img, i))
            #     overlay.save(overlay_file)
            # print('[{}] -- {} processed'.format(count, current_organ))


# 4 --- Create montage and save to overlay folder
count = 0
for case in os.listdir(overlay_dir):
    count += 1
    case_dir = os.path.join(overlay_dir,case)
    print('Creating montage {}'.format(case))
    # all_organ_montage_file = os.path.join(case_dir, case + 'all_organs_montage.png')
    # if os.path.isfile(all_organ_montage_file):
    #     print('[{}] Existed, skiping {}'.format(count, case))
    #     continue
    
    for organ in os.listdir(case_dir):
        if organ == 'all_organs':
            organ_dir = os.path.join(case_dir, organ)
            if os.path.isdir(organ_dir):
                new_img = Image.new('RGB', (168*8,168*8))

                for img_idx in range(0, 64):
                    slice_name = case + '_{}.png'.format(img_idx)
                    try:
                        slice_file = os.path.join(organ_dir, slice_name)
                        image = Image.open(slice_file).convert('RGB')
                    except IOError:
                        image = np.zeros((168,168))
                        image = Image.fromarray(image).convert('RGB')
                        print('IO Error, {}'.format(img_idx))
                    new_img.paste(image, (np.mod((img_idx), 8) * 168, 168 * int((img_idx) / 8)))
                new_img_file = os.path.join(case_dir, case + '{}_montage.png'.format(organ))
                new_img.save(new_img_file)
                print('[{}] {} montaged'.format(count, organ))






# copy montage images to other folder

overlay_dir = ''
output_dir = ''

count = 0
for case in os.listdir(overlay_dir):
    count += 1
    case_dir = os.path.join(overlay_dir, case)
    all_organ_montage_file = os.path.join(case_dir, '{}all_organs_montage.png'.format(case))
    output_file = os.path.join(output_dir, '{}all_organs_montage.png'.format(case))
    if os.path.isfile(output_file):
        print('[{}] Exists, skip {}'.format(count, case))
        continue
    os.system('cp \"{}\" \"{}\"'.format(all_organ_montage_file, output_file))
    print('[{}] copied'.format(count))




# # --- Create single organ montage and save to overlay folder
count = 0
for case in os.listdir(overlay_dir):
    count += 1
    case_dir = os.path.join(overlay_dir,case)
    print('Creating montage {}'.format(case))

    if os.path.isdir(case_dir):
        new_img = Image.new('RGB', (168*8,168*8))

        for img_idx in range(0, 64):
            slice_name = case + '_{}.png'.format(img_idx)
            try:
                slice_file = os.path.join(case_dir, slice_name)
                image = Image.open(slice_file).convert('RGB')
            except IOError:
                image = np.zeros((168,168))
                image = Image.fromarray(image).convert('RGB')
                print('IO Error, {}'.format(img_idx))
            new_img.paste(image, (np.mod((img_idx), 8) * 168, 168 * int((img_idx) / 8)))
        new_img_file = os.path.join(case_dir, case + '_montage.png')
        new_img.save(new_img_file)
        print('[{}] montaged'.format(count))

