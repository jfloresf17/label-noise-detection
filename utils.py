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
            
    
