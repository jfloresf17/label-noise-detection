import cv2
import numpy as np
from pathlib import Path

# Directories of the image datasets
input_path = Path('/media/tidop/Datos_4TB/databases/whu_resized')

image_files = sorted(list(input_path.glob('*/Image/*.png')))
label_files = sorted(list(input_path.glob('*/Mask/*.png')))

print(f'Number of images: {len(image_files)}')

# Create a directory to store the harmonized images
output_path = input_path.parent / 'harmonized_whu'
output_path.mkdir(exist_ok=True)

## Get the folder dataset to harmonize
target_folder = Path('/media/tidop/Datos_4TB/databases/kaggle/dataset/training_patches')
target_files = sorted(list(target_folder.glob('*.png')))
 

# Calculate global statistics for dataset B
mean_B = []
std_B = []
for i, image_path in enumerate(target_files):
    image_B = cv2.imread(str(image_path))  # Load as RGB
    mean_B.append(image_B.mean(axis=(0, 1)))  # Mean for each channel
    std_B.append(image_B.std(axis=(0, 1)))  # Std for each channel

    print(f'[{i+1}/{len(target_files)}]: {image_path.stem} done!')


mean_B = np.mean(mean_B)
std_B = np.mean(std_B)

# Harmonize images in dataset A using a linear transformation
for i, (image_path, label_path) in enumerate(zip(image_files, label_files)):
    ## Open the image
    image_A = cv2.imread(str(image_path))

    ## Open the label
    label_A = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    
    # Calculate mean and standard deviation of image A for each channel
    mean_A = image_A.mean(axis=(0, 1))
    std_A = image_A.std(axis=(0, 1))
    
    # Apply linear transformation to harmonize with dataset B for each channel
    harmonized_image = ((image_A - mean_A) / std_A) * std_B + mean_B
    harmonized_image = np.clip(harmonized_image, 0, 255).astype(np.uint8)
    
    # Save the harmonized image
    filename = image_path.stem
    folder_name = image_path.parent.parent.stem
    folder = output_path / folder_name
    folder.mkdir(exist_ok=True)

    input_folder = folder / 'Image'
    input_folder.mkdir(exist_ok=True)

    label_folder = folder / 'Mask'
    label_folder.mkdir(exist_ok=True)

    cv2.imwrite(input_folder / f'{filename}.png', harmonized_image)
    cv2.imwrite(label_folder / f'{filename}.png', label_A)

    print(f'[{i+1}/{len(image_files)}]: {image_path.stem} done!')