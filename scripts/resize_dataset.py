## WHU Building Dataset has 8,188 images with 512x512x3 dimensions 
## and their corresponding labels: {0: 'not building', 1: 'building'}
import pathlib
import cv2
import numpy as np


## How is probed SR algorithms (specifically, s2dr3)?
## Divide dataset to 512x512x3 images for 256x256x3 images
### 1. Load the dataset
path = pathlib.Path('/media/tidop/Datos_4TB/databases/whu')

image_files = sorted(list(path.glob('*/Image/*.png')))
label_files = sorted(list(path.glob('*/Mask/*.png')))

print(f'Number of images: {len(image_files)}')


### 2. Resize the images in 256x256xn dimensions
resized_folder = path.parent / 'whu_resized'
resized_folder.mkdir(exist_ok=True)

## Create folders
for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    ## Using cv2
    image = cv2.imread(str(image_file))
    label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

    filename = image_file.stem
    folder_name = image_file.parent.parent.stem
    folder = resized_folder / folder_name
    folder.mkdir(exist_ok=True)

    input_folder = folder / 'Image'
    input_folder.mkdir(exist_ok=True)

    label_folder = folder / 'Mask'
    label_folder.mkdir(exist_ok=True)

    ## Resize the image and label using cv2
    image_resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, (256, 256), interpolation=cv2.INTER_LINEAR)
    label_resized = (label_resized > 128).astype(np.uint8)

    ## Save the resized images
    cv2.imwrite(str(input_folder / f'{filename}.png'), image_resized)
    cv2.imwrite(str(label_folder / f'{filename}.png'), label_resized)

    print(f'[{i+1}/{len(image_files)}]: {image_file.stem} done!')
