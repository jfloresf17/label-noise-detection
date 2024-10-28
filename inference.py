## Make inference from Student model
import torch
import cv2
import pathlib

from models.unet_model import UNetStudent
from torchvision import transforms

def preprocess_image_for_inference(image_path, normalize=False, mean=None, std=None):
    # Lee la imagen
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR).astype(float)

    # Define la pipeline de preprocesamiento
    preprocess_pipeline = transforms.Compose([
        transforms.ToTensor()  # Convierte la imagen a tensor
    ])

    if normalize and mean is not None and std is not None:
        preprocess_pipeline.transforms.append(transforms.Normalize(mean=mean, std=std))

    # Aplica el preprocesamiento
    image = preprocess_pipeline(image).float()
    image = image.unsqueeze(0)  # Añade la dimensión de batch

    return image

def inference_student(checkpoint_path, normalize, mean, std, path_to_image, threshold, output_path):
    # Load the Student model
    student = UNetStudent(n_channels=3, n_classes=1)
    checkpoint = torch.load(checkpoint_path)
    filter_ckpt = {k: v for k, v in checkpoint.items()}
    student.load_state_dict(filter_ckpt)
    student = student.cuda()
    student.eval()

    # Load the image
    path_to_image = pathlib.Path(path_to_image) 
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = path_to_image.name
    output_filename = output_path / filename

    image = preprocess_image_for_inference(path_to_image, normalize=normalize, 
                                           mean=mean, std=std).cuda()

                
    with torch.no_grad():
        logits, _ = student(image)
        output = torch.sigmoid(logits) # To get the probabilities
        output = (output > threshold).float() ## Binarize the output

    ## Save the output image
    output = output.cpu().squeeze().numpy()
    ##  Using cv2
    cv2.imwrite(str(output_filename), (output).astype(int))


path = "/media/tidop/Datos_4TB/databases/kaggle/dataset/training_patches"
image_files = sorted(list(pathlib.Path(path).glob("*.png")))
checkpoint_path = "checkpoints/student_best_model.pth"
threshold = 0.5
output_path = "/media/tidop/Datos_4TB/databases/kaggle/dataset/output"
normalize = True
mean = [72.74413315, 99.76137101, 82.70024275] 
std = [36.28290664, 34.82507359, 41.48902725]

for i, image_file in enumerate(image_files):
    inference_student(checkpoint_path, normalize, mean, std, image_file, threshold, output_path)
    print(f"Processed image {i+1}/{len(image_files)}")

## Apply IoU
from scores import weighted_iou as wiou
import torch
import cv2
import pathlib
import numpy as np
import pandas as pd

label_path = "/media/tidop/Datos_4TB/databases/kaggle/dataset/training_noisy_labels"
output_path = "/media/tidop/Datos_4TB/databases/kaggle/dataset/output"
noisy_files = sorted(list(pathlib.Path(label_path).glob("*.png")))
pred_files = sorted(list(pathlib.Path(output_path).glob("*.png")))

noise_scores = []
for i, (noisy, pred) in enumerate(zip(noisy_files, pred_files)):
    filename = noisy.name
    noisy_image = cv2.imread(str(noisy), cv2.IMREAD_GRAYSCALE)
    pred_image = cv2.imread(str(pred), cv2.IMREAD_GRAYSCALE)

    ## Flatten the images
    noisy_image = torch.from_numpy(noisy_image)
    pred_image = torch.from_numpy(pred_image)

    ## Apply IoU
    iou_score = wiou(noisy_image, pred_image)
    noise_scores.append([filename, iou_score.numpy()])

    print(f"[{i+1}/{len(noisy_files)}] IoU Score for {filename}: {iou_score}")

## Save the results

df = pd.DataFrame(noise_scores, columns=["imageid", "Noise Score"])

## Order by IoU Score
df = df.sort_values(by="Noise Score", ascending=True) 
df["id"] = range(0, len(df))

df[['id', 'imageid']].to_csv("noise_scores.csv", index=False)


df[df["IoU Score"] == 0]