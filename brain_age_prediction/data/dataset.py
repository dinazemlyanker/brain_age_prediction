import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd

class BaDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, degrade=True):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing the 'path' and 'age' columns
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.degrade = degrade
 
    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.dataframe)
 
    def __getitem__(self, idx):
        """Returns a single sample (image, label) from the dataset"""
        row = self.dataframe.iloc[idx]
        image_path = row['preprocessed_path']
        age = row['age']
        # Load the NIfTI image
        image = self.load_nifti(image_path)
        image = image.reshape(1,160,160,192)
        # Apply transformation if provided
        if self.degrade:
            image = degrade_nifti(image)
 
        return image, torch.tensor(age, dtype=torch.float32)
 
    def load_nifti(self, path: str) -> np.ndarray:
        """Load a NIfTI image from the given path"""
        img = nib.load(path)
        img_data = img.get_fdata()  # get the numpy array of the image data
        img_data = np.expand_dims(img_data, axis=-1)  # Add channel dimension
        return img_data

class BaInferenceDataset(BaDataset):
    def __getitem__(self, idx):
        """Returns a single sample image from the dataset"""
        row = self.dataframe.iloc[idx]
        image_path = row['preprocessed_path']
        # Load the NIfTI image
        image = self.load_nifti(image_path)
        image = image.reshape(1,160,160,192)
        image = torch.from_numpy(image).float()
 
        return image