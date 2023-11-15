"""
For more info on datasets and dataloading, go to:
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloadersimport sys
"""

# import method from this site: https://www.geeksforgeeks.org/python-import-from-parent-directory/
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)
 
# now we can import the module in the parent directory
# from elpv.utils.elpv_reader import load_dataset

import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from PIL import Image
import pandas as pd

class ELPVImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        annotations_file: file
            csv file containing the image pathname and corresponding labels
        
        img_dir: str
            path to the elpv dataset folder
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.img_labels.at[idx, 'path'])

        # image = read_image(path, mode=ImageReadMode.RGB)
        # convert uint8 tensor to float tensor, then normalize 
        # image = image.float() / 255.0

        image = Image.open(path).convert("RGB")
        
        
        cell_type = self.img_labels.at[idx, 'type']
        defect_probability = self.img_labels.at[idx, 'probability']
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        
        return image, cell_type, defect_probability