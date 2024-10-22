## Calculate the mean and standard deviation of the training data
from pathlib import Path
import yaml
import cv2

def calculate_mean_std(train_path):
    # Load training data
    train_dir = Path(train_path)
    image_paths = list(train_dir.glob('*.png'))

    # Calculate mean and std    
    mean = [0, 0, 0]
    std = [0, 0, 0]

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))
        print(f"{i+1}/{len(image_paths)}", end='\r')
    
    mean /= len(image_paths)
    std /= len(image_paths)

    return mean, std

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        
        except yaml.YAMLError as exc:
            print(exc)
            
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()

        # Features extractor part (same as before)
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Final 1x1 convolution to generate num_classes output channels (probability map for each class)
        self.conv_out = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        # Pass the input through the feature extractor part
        x = self.features(x)

        # Apply 1x1 convolution to generate a feature map of size (num_classes, H, W)
        x = self.conv_out(x)

        # Apply softmax across the channel dimension (dim=1), to get probabilities for each class
        x = F.softmax(x, dim=1)

        # Optionally: If you want the output to match the original input size, apply interpolation
        # x = F.interpolate(x, size=(original_height, original_width), mode='bilinear', align_corners=False)

        return x


import cv2
image = cv2.imread("/media/tidop/Datos_4TB/databases/whu/train/Mask/train_0003.png")
image = image

### With compose
import torch
from torchvision import transforms
image_transforms = transforms.Compose([
            ## Convert to tensor
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),  # Resize to 256x256
            transforms.Normalize(mean=[105.34253814, 114.2284708 , 112.52936415], 
                                 std=[42.12451161, 39.52149692, 42.81161886])
])

transf_image = image_transforms(image)
transf_image.shape
print(transf_image.min(), transf_image.max())


## Without compose
image2 = cv2.imread("/media/tidop/Datos_4TB/databases/whu_resized/train/Image/train_0003.png", cv2.IMREAD_COLOR)
image2 = image2.astype(float)

image_tensor = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

mean_tensor = torch.tensor([105.34253814, 114.2284708 , 112.52936415], dtype=torch.float32).view(-1, 1, 1)
std_tensor = torch.tensor([42.12451161, 39.52149692, 42.81161886], dtype=torch.float32).view(-1, 1, 1)

transf_image2 = (image_tensor - mean_tensor) / std_tensor
transf_image2.shape

print(transf_image2.min(), transf_image2.max())


## Plot the transformed image
import matplotlib.pyplot as plt
plt.imshow(transf_image.permute(1, 2, 0))
plt.axis('off')
plt.savefig("transformed_image.png", bbox_inches='tight')
plt.close()
plt.clf()
