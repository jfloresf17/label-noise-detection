import cv2
import pathlib
import yaml

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_mean_std(image_paths):
    mean = 0.
    std = 0.
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        mean += img.mean(axis=(0,1))
        std += img.std(axis=(0,1))
        print(f"Processed {i+1}/{len(image_paths)} images", end='\r')

    mean /= len(image_paths)
    std /= len(image_paths)
    return mean, std

def cut_raster(raster_files, output_path, tile_size):

    ## Create the output directory if it does not exist
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    for n, raster_path in enumerate(raster_files):
        ## Using cv2 to read the image
        img = cv2.imread(raster_path, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        ## Calculate the number of tiles in each dimension
        num_tiles_x = width // (tile_size)
        num_tiles_y = height // (tile_size)

        ## Generate the tiles
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                start_x = i * (tile_size)
                start_y = j * (tile_size)
                end_x = start_x + tile_size
                end_y = start_y + tile_size

                ## Extract the tile
                tile = img[start_y:end_y, start_x:end_x, :]

                filename = raster_path.stem
                cv2.imwrite(f"{output_path}/{filename}_{i}_{j}.png", tile)

        print(f"Processed {n+1}/{len(raster_files)} images", end='\r')