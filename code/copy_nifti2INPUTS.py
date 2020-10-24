import os

imagedir = ''
output_dir = ''
count = 0
for subject in os.listdir(imagedir):
    subject_dir = os.path.join(imagedir, subject)
    if subject.startswith('TumaControl') and os.path.isdir(subject_dir):
        subject_index = int(subject.split('_')[1])
        if subject_index <= 600:
            for fold_name in os.listdir(subject_dir):
                folder_dir = os.path.join(subject_dir, fold_name)
                if os.path.isdir(folder_dir):
                    for nii_name in os.listdir(folder_dir):
                        if nii_name.endswith('.nii'):
                            nii_file = os.path.join(folder_dir, nii_name)
                            output_file = os.path.join(output_dir, nii_name)
                            count += 1
                            if os.path.isfile(output_file):
                                print('[{}] Skip {}'.format(count, nii_name))
                                continue
                            os.system('cp \"{}\" \"{}\"'.format(nii_file, output_file))
                            print('[{}] COpoed {}'.format(count, nii_name))

# check number of slices
