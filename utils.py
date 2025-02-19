import cv2
import pathlib
import yaml
import numpy as np

def load_config(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def cut_raster(raster_files, output_path, tile_size):
    """
    Cut raster images into smaller tiles.

    Args:
        raster_files (list): List of paths to raster images.
        output_path (str): Directory to save the tiles.
        tile_size (int): Size of each tile.

    Returns:
        None
    """
    # Create the output directory if it does not exist
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    for n, raster_path in enumerate(raster_files):
        # Using cv2 to read the image
        img = cv2.imread(raster_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        # Calculate the number of tiles in each dimension
        num_tiles_x = width // tile_size
        num_tiles_y = height // tile_size

        # Generate the tiles
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                start_x = i * tile_size
                start_y = j * tile_size
                end_x = start_x + tile_size
                end_y = start_y + tile_size

                # Extract the tile
                tile = img[start_y:end_y, start_x:end_x, :]

                filename = raster_path.stem
                cv2.imwrite(f"{output_path}/{filename}_{i}_{j}.png", tile)

        print(f"Processed {n+1}/{len(raster_files)} images", end='\r')


def calculate_mean_and_std(image_paths):
    """
    Calculate the mean and standard deviation of all provided images.

    Args:
        image_paths (list): List of paths to images.

    Returns:
        tuple: Mean and standard deviation of the images.
    """
    pixel_sum = np.zeros(3)
    pixel_squared_sum = np.zeros(3)
    num_pixels = 0

    for i, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        # Update the pixel sum and pixel squared sum
        pixel_sum += np.sum(image, axis=(0, 1))
        pixel_squared_sum += np.sum(image ** 2, axis=(0, 1))
        num_pixels += image.shape[0] * image.shape[1]

        print(f"Processed {i+1}/{len(image_paths)} images", end='\r')

    # Calculate the mean and standard deviation
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))
    return mean, std


def normalize_dataset(input_dir, label_dir, target_mean, target_std, output_dir):
    """
    Adjust the mean and standard deviation of a dataset to the given target.

    Args:
        input_dir (list): List of paths to input images.
        label_dir (list): List of paths to label images.
        target_mean (np.array): Target mean for normalization.
        target_std (np.array): Target standard deviation for normalization.
        output_dir (str): Directory to save the normalized images.

    Returns:
        None
    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for i, (img_path, label_path) in enumerate(zip(input_dir, label_dir)):

        # Read the input image
        input_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)

        # Read the label image
        label_image = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        # Calculate the mean and standard deviation of the input image
        input_mean = np.mean(input_image, axis=(0, 1))
        input_std = np.std(input_image, axis=(0, 1))

        # Normalize the input image
        normalized_image = (input_image - input_mean) / (input_std + 1e-8)  # Z-score normalization
        # Scale to the target mean and standard deviation
        scaled_image = (normalized_image * target_std) + target_mean

        # Ensure values are in the range [0, 255]
        scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)

        # Convert from RGB to BGR for saving using cv2
        scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)

        # Save the adjusted image in the output directory
        output_path = output_dir / f"{img_path.parent.parent.name}/Image"
        output2_path = output_dir / f"{label_path.parent.parent.name}/Mask"
        output_path.mkdir(parents=True, exist_ok=True)
        output2_path.mkdir(parents=True, exist_ok=True)

        output_imagefile = output_path / img_path.name
        output_label = output2_path / label_path.name
        print(output_path)

        cv2.imwrite(str(output_imagefile), scaled_image)
        cv2.imwrite(str(output_label), label_image)

        print(f"Processed {i+1}/{len(input_dir)} images", end='\r')