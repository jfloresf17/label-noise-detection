import torch
from models.teacher_unet import TeacherUNetModel
from pathlib import Path
import cv2
import numpy as np
from utils import load_yaml

def generate_teacher_outputs(config_path: str="config.yaml"):
    # Load configuration
    config = load_yaml(config_path)
    # Load pre-trained teacher model
    teacher_model = TeacherUNetModel(config)
    teacher_model.load_state_dict(torch.load("checkpoints/best_teacher_notnorm.ckpt")["state_dict"])
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

        if config["Normalize"]["apply"]:
            ## Convert list to tensor
            mean = torch.tensor(config["Normalize"]["WHU"]["mean"], 
                                dtype=torch.float32).view(-1, 1, 1)

            std = torch.tensor(config["Normalize"]["WHU"]["std"],
                            dtype=torch.float32).view(-1, 1, 1)

            image = (image - mean) / std

        # Generate output from teacher model
        with torch.no_grad():
            output = teacher_model(image)
        
        # Convert output to numpy and save as PNG
        output_np = output.squeeze().cpu().numpy()
        output_np = output_np.astype(np.uint8)
        output_filename = output_dir / image_path.name
        cv2.imwrite(str(output_filename), output_np)
    
        print(f"{i+1}/{len(image_paths)}", end='\r')

## Generate teacher outputs
generate_teacher_outputs(config_path="config.yaml")

## TODO: Adjust histogram matching for teacher outputs with match_histograms() function of scikit-image

