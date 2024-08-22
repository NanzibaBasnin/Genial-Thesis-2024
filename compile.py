import nibabel as nib
import cv2
import numpy as np
import os

def jpg_to_nifti(jpg_folder, nifti_folder):
    if not os.path.exists(nifti_folder):
        os.makedirs(nifti_folder)
    for filename in os.listdir(jpg_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(jpg_folder, filename), cv2.IMREAD_GRAYSCALE)
            img_nifti = np.stack([img]*33, axis=-1)  # Creating a dummy 3D stack
            nifti_file = nib.Nifti1Image(img_nifti, affine=np.eye(4))
            nib.save(nifti_file, os.path.join(nifti_folder, filename.replace('.jpg', '.nii')))
