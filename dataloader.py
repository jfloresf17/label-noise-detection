import torch
from torch.utils.data import DataLoader, Dataset
import pathlib
import pytorch_lightning as pl
import numpy as np
import cv2
from utils import load_yaml
from sklearn.model_selection import train_test_split

# Here we define the datasets and dataloaders for the teacher and student models
class TeacherDataset(Dataset):
    def __init__(self, image_paths, label_paths, normalize=False, mean=None, std=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image =  cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.label_paths[idx]), cv2.IMREAD_GRAYSCALE)

        ## Convert to tensor
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)

        if self.normalize:
            ## Normalize the image
            mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
            std_tensor = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)

            image = (image - mean_tensor) / std_tensor
        
    
        return image, label
    
class WHUDataModule(pl.LightningDataModule):
    def __init__(self, file_paths, normalize, mean, std, batch_size=32, num_workers=4):
        super().__init__()
        self.file_paths = file_paths
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        ## From training folder
        train_input_files = sorted(list((self.file_paths / "train/Image").glob('*.png')))
        train_label_files = sorted(list((self.file_paths / "train/Mask").glob('*.png')))
        
        ## From validation folder
        val_input_files = sorted(list((self.file_paths / "val/Image").glob('*.png')))
        val_label_files = sorted(list((self.file_paths / "val/Mask").glob('*.png')))

        ## From test folder
        test_input_files = sorted(list((self.file_paths / "test/Image").glob('*.png')))
        test_label_files = sorted(list((self.file_paths / "test/Mask").glob('*.png')))
    
        if stage == 'fit' or stage is None:
            self.train_dataset = TeacherDataset(train_input_files, train_label_files, 
                                                normalize=self.normalize, mean=self.mean, 
                                                std=self.std)
            
            self.val_dataset = TeacherDataset(val_input_files, val_label_files,
                                              normalize=self.normalize,mean=self.mean, 
                                              std=self.std)
            
        if stage == 'test' or stage == 'predict':
            self.test_dataset = TeacherDataset(test_input_files, test_label_files,
                                               normalize=self.normalize, mean=self.mean, 
                                               std=self.std)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False)
    

class StudentDataset(Dataset):
    def __init__(self, image_paths, label_paths, teacher_output_paths, 
                 normalize=False, mean=None, std=None):
        
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.teacher_output_paths = teacher_output_paths
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.label_paths[idx]), cv2.IMREAD_GRAYSCALE)
        teacher_output = cv2.imread(str(self.teacher_output_paths[idx]), cv2.IMREAD_GRAYSCALE) 
        
        ## Convert to tensor
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        teacher_output = torch.tensor(teacher_output, dtype=torch.float).unsqueeze(0)

        if self.normalize:
            ## Normalize the image
            mean_tensor = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
            std_tensor = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)

            image = (image - mean_tensor) / std_tensor

        return image, label, teacher_output
    
class DataCentricDataModule(pl.LightningDataModule):
    def __init__(self, input_files, label_files, teacher_output_files, normalize,
                 mean, std, batch_size=32, num_workers=4):
        super().__init__()
        self.input_files = input_files
        self.label_files = label_files
        self.teacher_output_files = teacher_output_files
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ## Zip the input, label and teacher output files
        files = list(zip(self.input_files, self.label_files, self.teacher_output_files))

        ## Split the dataset using sklearn
        ttrain_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(ttrain_files, test_size=0.2, random_state=42)

        ## Unzip the files
        train_input_files, train_label_files, train_teacher_files = zip(*train_files)
        val_input_files, val_label_files, val_teacher_files = zip(*val_files)
        test_input_files, test_label_files, test_teacher_files = zip(*test_files)

   
        if stage == 'fit' or stage is None:
            self.train_dataset = StudentDataset(train_input_files, train_label_files, 
                                                train_teacher_files, self.normalize,
                                                self.mean, self.std)
            
            self.val_dataset = StudentDataset(val_input_files, val_label_files,
                                              val_teacher_files, self.normalize,
                                              self.mean, self.std)
        
        if stage == 'test' or stage == 'predict':
            self.test_dataset = StudentDataset(test_input_files, test_label_files,
                                               test_teacher_files, self.normalize,
                                               self.mean, self.std)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          shuffle=False)
