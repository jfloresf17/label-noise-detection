import torch
from models.teacher_unet import TeacherUNetModel
from pathlib import Path
import cv2
import numpy as np
from utils import load_yaml
from torchvision import transforms

def generate_teacher_outputs(config_path: str="config.yaml"):
    # Load configuration
    config = load_yaml(config_path)
    # Load pre-trained teacher model
    teacher_model = TeacherUNetModel(config)
    teacher_model.load_state_dict(torch.load("checkpoints/best_teacher.ckpt")["state_dict"])
    teacher_model.eval()
    
    # Directory for teacher outputs
    output_dir = Path(config["data"]["teacher_output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load images and generate outputs
    image_dir = Path(config["data"]["datacentric_image_path"])
    image_paths = list(image_dir.glob('*.png'))
    
    for i, image_path in enumerate(image_paths):
        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        ## Convert to tensor
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1).unsqueeze(0)



        # Define a transformation pipeline for the image
        image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ])

        # Add normalization if requested
        if config["Normalize"]["apply"]:
            image_transforms.transforms.append(
                transforms.Normalize(mean=config["Normalize"]["WHU"]["mean"], 
                                     std=config["Normalize"]["WHU"]["std"])  # Normalize image
            )

        # Read image and label using cv2
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        # Convert the images to float
        image = image.astype(float)

        # Apply the transformations
        image = image_transforms(image).float()

        with torch.no_grad():
            output = teacher_model(image[None])
        
        # Convert output to numpy and save as PNG
        output_np = output.squeeze().cpu().numpy()
        output_np = output_np.astype(np.uint8) / 255.0
        output_filename = output_dir / image_path.name
        cv2.imwrite(str(output_filename), output_np)
    
        print(f"{i+1}/{len(image_paths)}", end='\r')

## Generate teacher outputs
generate_teacher_outputs(config_path="config.yaml")

## TODO: Adjust histogram matching for teacher outputs with match_histograms() function of scikit-image

