import cv2
import pytorch_lightning as pl
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class TeacherDataset(Dataset):
    def __init__(self, image_paths, label_paths, normalize, mean=None, std=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # Define a transformation pipeline for the image
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        # Define a transformation pipeline for the label (no normalization required)
        self.label_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
        ])

        # Add normalization if requested
        if self.normalize:
            self.image_transforms.transforms.append(
                transforms.Normalize(mean=self.mean, std=self.std)  # Normalize image
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image and label using cv2
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.label_paths[idx]), cv2.IMREAD_GRAYSCALE)

        # Convert the images to float
        image = image.astype(float)
        label = label.astype(float) / 255.0

        # Apply the transformations
        image = self.image_transforms(image).float()
        label = self.label_transforms(label).float()  # Apply label transformations (resize, tensor)

        return image, label
    
    
class WHUDataModule(pl.LightningDataModule):
    def __init__(self, file_paths, normalize, mean, std, batch_size=32, num_workers=4):
        super().__init__()
        self.file_paths = pathlib.Path(file_paths)
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
    def __init__(self, image_paths, label_paths, 
                 normalize, mean=None, std=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # Define the preprocessing pipeline
        self.image_preprocess_pipeline = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensor 
        ])

        # Define the label preprocessing pipeline
        self.label_preprocess_pipeline = transforms.Compose([
            transforms.ToTensor(),  # Convert labels to tensor
        ])        

        # If normalization is required, add it to the pipeline
        if self.normalize and self.mean is not None and self.std is not None:
            self.image_preprocess_pipeline.transforms.append(
                transforms.Normalize(mean=self.mean, std=self.std)  # Normalize images to [0, 1]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        label = cv2.imread(str(self.label_paths[idx]), cv2.IMREAD_GRAYSCALE)

        image = image.astype(float)
        label = label.astype(float)

        # Apply preprocessing pipeline
        image = self.image_preprocess_pipeline(image).float()
        label = self.label_preprocess_pipeline(label).float()

        return image, label


class DataCentricDataModule(pl.LightningDataModule):
    def __init__(self, input_files, label_files, normalize,
                 mean, std, batch_size=32, num_workers=4):
        super().__init__()
        self.input_files = input_files
        self.label_files = label_files
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ## Set the seed for reproducibility
        torch.manual_seed(42)

        ## Zip the input, label and teacher output files
        files = list(zip(self.input_files, self.label_files))

        ## Split the dataset using sklearn
        ttrain_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(ttrain_files, test_size=0.2, random_state=42)

        ## Unzip the files
        train_input_files, train_label_files = zip(*train_files)
        val_input_files, val_label_files = zip(*val_files)
        test_input_files, test_label_files = zip(*test_files)

        if stage == 'fit' or stage is None:
            self.train_dataset = StudentDataset(train_input_files, train_label_files, 
                                                self.normalize, self.mean, self.std)
            
            self.val_dataset = StudentDataset(val_input_files, val_label_files,
                                              self.normalize, self.mean, self.std)
        
        if stage == 'test' or stage == 'predict':
            self.test_dataset = StudentDataset(test_input_files, test_label_files,
                                               self.normalize, self.mean, self.std)
        
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