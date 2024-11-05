import torch
import torch.nn as nn
import wandb
import random
import pathlib
import typer
import numpy as np

from loss_function import LovaszLossBatch
from utils import load_config
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.unet_model import ResUnetTeacher, ResUnetStudent
from dataloader import DataCentricDataModule
from scores import iou, dice_coefficient, precision, recall, f1_score

def student_teacher(config_path: str):
    
    # Load configuration from YAML file
    config = load_config(config_path)

    input_paths = config["data"]["datacentric_image_path"]
    label_paths = config["data"]["label_noisy_path"]
    normalize = config['Normalize']["apply"]
    mean = config['Normalize']["DataCentric"]["mean"]
    std = config['Normalize']["DataCentric"]["std"]
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    lr = float(config["learning_rate"])
    alpha = float(config["knowledge_distillation"]["alpha"])
    beta = float(config["knowledge_distillation"]["beta"])
    theta = float(config["knowledge_distillation"]["theta"])
    epochs = config["trainer"]["max_epochs"]
    device = config["device"]

    # Initialize Weights & Biases
    wandb.init(project=config["trainer"]["wandb_project"], config=config, 
               name=config["trainer"]["experiment_name"])

    teacher = ResUnetTeacher(channel=3)  # Teacher model
    teacher.load_state_dict(torch.load("checkpoints/teacher_alabama_resunet_best_model.pth"))
    teacher.to(device)
    student = ResUnetStudent(channel=3).to(device)  # Student model

    # Create the data module
    input_files = sorted(list(pathlib.Path(input_paths).glob("*.png")))
    label_files = sorted(list(pathlib.Path(label_paths).glob("*.png")))
                         
    data_module = DataCentricDataModule(input_files, label_files, normalize, mean, 
                                        std, batch_size=batch_size, 
                                        num_workers=num_workers)
    data_module.setup(stage='fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    

    # The teacher is set to evaluation mode and its parameters are frozen
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Optimizer and Loss Functions
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    criterion_ce = nn.BCELoss().to(device)  # Binary Cross-Entropy Loss for binary segmentation
    
    # MSE loss for distillation
    criterion_mse = nn.MSELoss().to(device)

    # Training with Distillation
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_precision = 0.0
        running_recall = 0.0
        running_f1 = 0.0

        # Training phase
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass for Teacher
            with torch.no_grad():
                t_logits, t_feature_map = teacher(images)

            s_logits, s_feature_map = student(images)

            # Calculate the loss using the feature maps: student and teacher
            mse_loss = criterion_mse(s_feature_map, t_feature_map)

            # Calculate the Lovasz loss: student logits and teacher hard labels
            t_hard_label = t_logits.float()  # Binarize the logits
            lovasz_l = LovaszLossBatch(s_logits, t_hard_label)

            # Calculate the Cross-Entropy loss
            s_prob = torch.sigmoid(s_logits)
            bce_loss =  criterion_ce(s_prob, targets)

            # Calculate the loss using the feature maps and the logits
            # loss_ce = criterion_ce(s_logits, targets)

            # mse_loss = torch.sum(s_logits  * (t_logits.log() - s_logits.log()))/ s_logits.size()[0] * 4          

            # Total loss combines the distillation and the CE loss
            loss = alpha * mse_loss + beta * bce_loss + theta * lovasz_l

            # Backward propagation and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate metrics
            batch_iou = iou(s_prob, targets)
            batch_dice = dice_coefficient(s_prob, targets)
            batch_precision = precision(s_prob, targets)
            batch_recall = recall(s_prob, targets)
            batch_f1 = f1_score(s_prob, targets)
           
            running_iou += batch_iou.item()
            running_dice += batch_dice.item()
            running_precision += batch_precision.item()
            running_recall += batch_recall.item()
            running_f1 += batch_f1.item()

        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_iou = running_iou / len(train_loader)
        epoch_dice = running_dice / len(train_loader)
        epoch_precision = running_precision / len(train_loader)
        epoch_recall = running_recall / len(train_loader)
        epoch_f1 = running_f1 / len(train_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], CE + MSE Loss: {epoch_loss:.4f}")

        # Log epoch metrics to WandB
        wandb.log({
            "Train CE + MSE Loss": epoch_loss,
            "Train IoU": epoch_iou,
            "Train Dice": epoch_dice,
            "Train Precision": epoch_precision,
            "Train Recall": epoch_recall,
            "Train F1": epoch_f1
        })

        # Validation phase
        student.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.cuda(), targets.cuda()
                logits, _ = student(images)
                prob = torch.sigmoid(logits)

                # Calculate the Cross-Entropy loss
                loss_ce = criterion_ce(prob, targets)
                val_loss += loss_ce.item()

                # Calculate validation metrics
                val_iou += iou(prob, targets).item()
                val_dice += dice_coefficient(prob, targets).item()
                val_precision += precision(prob, targets).item()
                val_recall += recall(prob, targets).item()
                val_f1 += f1_score(prob, targets).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)

        print(f"Validation CE + MSE Loss: {val_loss:.4f}",
              f"Validation IoU: {val_iou:.4f}",
              f"Validation Dice: {val_dice:.4f}",
              f"Validation Precision: {val_precision:.4f}",
              f"Validation Recall: {val_recall:.4f}",
              f"Validation F1: {val_f1:.4f}")

        # Log validation metrics to WandB
        wandb.log({
            "Validation CE + MSE Loss": val_loss,
            "Validation IoU": val_iou,
            "Validation Dice": val_dice,
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1": val_f1
        })

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Obtén un lote de validación y pasa las imágenes por el modelo
                sample_images, sample_masks = next(iter(val_loader))
                sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)
                sample_outputs, _ = student(sample_images)
                sample_outputs = (torch.sigmoid(sample_outputs) > 0.5).float()  # Binarizar predicciones

                # Seleccionar una imagen aleatoria del lote
                random_index = random.randint(0, sample_images.size(0) - 1)
                
                wandb.log({
                            "Validation Example": wandb.Image(
                                sample_images[random_index].cpu(), 
                                caption="Image Example Kaggle",
                                masks={
                                    "predictions": {
                                        "mask_data": sample_outputs[random_index][0].cpu().numpy(),
                                        "class_labels": {0: "no building", 1: "building"}
                                    },
                                    "ground_truth": {
                                        "mask_data": sample_masks[random_index][0].cpu().numpy(),
                                        "class_labels": {0: "no building", 1: "building"}
                                    }
                                }
                            )
                        })

        # Adjust learning rate based on validation loss
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != current_lr:
            print(f"Learning rate changed to {new_lr}")

        
        # Switch back to training mode
        student.train()
        # Save the best model checkpoint (by validation IoU)
        if epoch == 0:
            max_ioU = val_iou
        if val_iou > max_ioU:
            max_ioU = val_iou
            torch.save(student.state_dict(), "checkpoints/student_resunet_best_model.pth")


    student.load_state_dict(torch.load("checkpoints/student_resunet_best_model.pth"))
    
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # Función para buscar el umbral óptimo en el conjunto de prueba
    def find_optimal_threshold(student, test_loader, device):
        thresholds = np.arange(0.1, 1.0, 0.1)
        best_threshold = 0.5
        best_metric = 0.0

        for threshold in thresholds:
            metric_total = 0.0
            count = 0

            with torch.no_grad():
                for images, targets in test_loader:
                    images, targets = images.to(device), targets.to(device)
                    outputs, _ = student(images)

                    outputs = (outputs > threshold).float()
                    metric_value = iou(outputs, targets).item()
                    metric_total += metric_value
                    count += 1

            average_metric = metric_total / count
            print(f"Threshold: {threshold}, Average IoU: {average_metric:.4f}")

            if average_metric > best_metric:
                best_metric = average_metric
                best_threshold = threshold

        print(f"Optimal Threshold: {best_threshold} with IoU: {best_metric:.4f}")
        return best_threshold

    # Cálculo del umbral óptimo al finalizar el entrenamiento
    optimal_threshold = find_optimal_threshold(student, test_loader, device)
    print(f"Optimal Threshold for Test Set: {optimal_threshold}")
    

if __name__ == "__main__":
    typer.run(student_teacher)

# To run the training script as a command line application in the background, use the following command:
# nohup python train_student.py config.yaml > logs/student_training.log 2>&1 &
