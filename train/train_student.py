import torch
import torch.nn as nn
import wandb
import pathlib
import typer
import yaml
from models.unet_model import UNetTeacher, UNetStudent
from dataloader import DataCentricDataModule
from scores import iou, dice_coefficient, precision, recall, f1_score

def load_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    epochs = config["trainer"]["max_epochs"]
    device = config["device"]

    # Initialize Weights & Biases
    wandb.init(project=config["wandb_project"], config=config, 
               name=config["experiment_name"])

    teacher = UNetTeacher(n_channels=3, n_classes=1)  # Pre-trained Teacher model
    teacher.load_state_dict(torch.load("checkpoints/teacher_best_model.pth"))
    teacher.to(device)
    student = UNetStudent(n_channels=3, n_classes=1).to(device)  # Student model

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

    criterion_ce = nn.BCEWithLogitsLoss().to(device)  # Binary Cross-Entropy Loss for binary segmentation
    criterion_mse = nn.MSELoss().to(device)  # Loss for feature distillation

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
                _, t_feature_map = teacher(images)
            
            s_logits, s_feature_map = student(images)

            # Calculate the loss using the feature maps and the logits
            mse_loss = criterion_mse(s_feature_map, t_feature_map)
            loss_ce = criterion_ce(s_logits, targets)

            # Total loss combines the distillation and the CE loss
            loss = alpha * mse_loss + beta * loss_ce

            # Backward propagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            batch_iou = iou(s_logits, targets)
            batch_dice = dice_coefficient(s_logits, targets)
            batch_precision = precision(s_logits, targets)
            batch_recall = recall(s_logits, targets)
            batch_f1 = f1_score(s_logits, targets)

            # Log results to WandB
            wandb.log(data={
                "Batch Loss": loss.item(),
                "Batch IoU": batch_iou.item(),
                "Batch Dice": batch_dice.item(),
                "Batch Precision": batch_precision.item(),
                "Batch Recall": batch_recall.item(),
                "Batch F1": batch_f1.item()
            }, step=epoch * len(train_loader) + len(train_loader))
            
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
            "Epoch CE + MSE Loss": epoch_loss,
            "Epoch IoU": epoch_iou,
            "Epoch Dice": epoch_dice,
            "Epoch Precision": epoch_precision,
            "Epoch Recall": epoch_recall,
            "Epoch F1": epoch_f1
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

                # Calculate the Cross-Entropy loss
                loss_ce = criterion_ce(logits, targets)
                val_loss += loss_ce.item()

                # Calculate validation metrics
                val_iou += iou(logits, targets).item()
                val_dice += dice_coefficient(logits, targets).item()
                val_precision += precision(logits, targets).item()
                val_recall += recall(logits, targets).item()
                val_f1 += f1_score(logits, targets).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        val_precision /= len(val_loader)
        val_recall /= len(val_loader)
        val_f1 /= len(val_loader)

        print(f"Validation Loss: {val_loss:.4f}",
              f"Validation IoU: {val_iou:.4f}",
              f"Validation Dice: {val_dice:.4f}",
              f"Validation Precision: {val_precision:.4f}",
              f"Validation Recall: {val_recall:.4f}",
              f"Validation F1: {val_f1:.4f}")

        # Log validation metrics to WandB
        wandb.log({
            "Validation Loss": val_loss,
            "Validation IoU": val_iou,
            "Validation Dice": val_dice,
            "Validation Precision": val_precision,
            "Validation Recall": val_recall,
            "Validation F1": val_f1
        })


        # Save the best model checkpoint
        if epoch == 0:
            min_val_loss = val_loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(student.state_dict(), "checkpoints/student_best_model.pth")

if __name__ == "__main__":
    typer.run(student_teacher)

# To run the training script as a command line application in the background, use the following command:
# nohup python train_student.py config.yaml > logs/student_training.log 2>&1 &
