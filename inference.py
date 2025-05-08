## Make inference from Student model
import torch
import cv2
import pathlib
import pandas as pd


from models.unet_model import NRNRSSEGStudent
from torchvision import transforms
from scores import weighted_iou as iou


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
    student = NRNRSSEGStudent(in_channels=3, out_channels=1, base_filters=32)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
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
        output, _ = student(image)
        output = (output > threshold).float() ## Binarize the output

    ## Save the output image
    output = output.cpu().squeeze().numpy()
    ##  Using cv2
    cv2.imwrite(str(output_filename), (output).astype(int))


path = "/media/tidop/Datos_4TB1/databases/full_dataset/dataset/training_patches"
image_files = sorted(list(pathlib.Path(path).glob("*.png")))
checkpoint_path = "/home/tidop/projects/Noisy-Student/checkpoints/student_nrnrsseg_sce+mse.pth"
threshold = 0.5
pred_path = "/media/tidop/Datos_4TB1/databases/full_dataset/dataset/inferenced_pred_labels"
normalize = True
mean = [72.74413315, 99.76137101, 82.70024275] 
std = [36.28290664, 34.82507359, 41.48902725]


for i, image_file in enumerate(image_files):
    inference_student(checkpoint_path, normalize, mean, std, image_file, threshold, pred_path)
    print(f"Processed image {i+1}/{len(image_files)}")

## Apply IoU
target_path = "/media/tidop/Datos_4TB1/databases/full_dataset/dataset/training_noisy_labels"

pred_files = sorted(list(pathlib.Path(pred_path).glob("*.png")))
target_files = sorted(list(pathlib.Path(target_path).glob("*.png")))

noise_scores = []
for i, (pred, target) in enumerate(zip(pred_files, target_files)):
    filename = pred.name
    pred_image = cv2.imread(str(pred), cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)

    ## Flatten the images
    pred_image = torch.from_numpy(pred_image)
    target_image = torch.from_numpy(target_image)

    ## Apply IoU
    iou_score = iou(target_image, pred_image)
    noise_scores.append([filename, iou_score.numpy()])

    print(f"[{i+1}/{len(target_files)}] IoU Score for {filename}: {iou_score}")

## Save the results
df = pd.DataFrame(noise_scores, columns=["imageid", "Noise Score"])

## Order by IoU Score
df = df.sort_values(by=["Noise Score"], ascending=False)
df["id"] = range(0, len(df))
df[['id', 'imageid']].to_csv("noise_scores.csv", index=False)
