import torch
import torch.nn as nn
import random
import torch.optim as optim
import wandb
import typer

from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss_function import DiceLoss
from utils import load_config
from torch.utils.data import ConcatDataset, DataLoader
from models.unet_model import  ResUnetTeacher
from dataloader import BuildingDataModule
from scores import iou, dice_coefficient, precision, recall, f1_score

def train_teacher(config_path: str):

    # Load configuration from YAML file
    config = load_config(config_path)

    normalize = config['Normalize']["apply"]
    
    whu_path = config["data"]["whu_path"]
    whu_mean = config['Normalize']["WHU"]['mean']
    whu_std = config['Normalize']["WHU"]['std']
    
    alabama_path = config["data"]["alabama_path"]       
    alabama_mean = config['Normalize']["Alabama"]['mean']
    alabama_std = config['Normalize']["Alabama"]['std']

    batch_size = config["data"]['batch_size']
    num_workers = config["data"]['num_workers']

    lr = float(config['learning_rate'])
    epochs = config["trainer"]['max_epochs']
    device = config["device"]

    # Initialize Weights & Biases
    wandb.init(project=config["trainer"]["wandb_project"], config=config, 
               name=config["trainer"]["experiment_name"])

    # Define the Teacher model
    teacher = ResUnetTeacher(channel=3)
    teacher = teacher.to(device)  # Use GPU if available

    # Define the optimizer and loss function
    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    bce_criterion = nn.BCELoss().to(device)  # For binary segmentation, we use Binary Cross Entropy
    criterion_dice = DiceLoss().to(device)

    scheduler = ReduceLROnPlateau(optimizer, # Reduce learning rate when a val loss metric has stopped improving 
                                  mode='min', # Minimize the loss
                                  factor=0.5, # Reduce the learning rate by half 
                                  patience=3 # Number of epochs with no improvement after which learning rate will be reduced 
                                  )

    # Create the data module for WHU and Alabama datasets
    whu_module = BuildingDataModule(whu_path, "WHU", normalize, whu_mean, whu_std, 
                                    batch_size=batch_size, num_workers=num_workers)
    whu_module.setup(stage='fit')

    whu_train_dataset = whu_module.train_dataloader().dataset
    whu_val_dataset = whu_module.val_dataloader().dataset

    alabama_module = BuildingDataModule(alabama_path, "Alabama", normalize, alabama_mean, alabama_std, 
                                        batch_size=batch_size, num_workers=num_workers)
    alabama_module.setup(stage='fit')

    alabama_train_dataset = alabama_module.train_dataloader().dataset
    alabama_val_dataset = alabama_module.val_dataloader().dataset


    # Concatenate the datasets for training and validation
    combined_train_dataset = ConcatDataset([whu_train_dataset, alabama_train_dataset])
    combined_val_dataset = ConcatDataset([whu_val_dataset, alabama_val_dataset])

    # Create the data loaders
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Train the Teacher model
    teacher.train()
    for epoch in range(epochs):
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        running_precision = 0.0
        running_recall = 0.0
        running_f1 = 0.0

        # Iterate over the training data
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            logits, _ = teacher(images)

            # Convert to probabilities
            outputs = torch.sigmoid(logits)

            # Calculate losses (BCE + Dice)
            bce_loss = bce_criterion(outputs, masks)
            dice_loss = criterion_dice(outputs, masks)

            ## Combine the losses
            loss = 0.5 * bce_loss + 0.5 * dice_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate metrics
            batch_iou = iou(outputs, masks)
            batch_dice = dice_coefficient(outputs, masks)
            batch_precision = precision(outputs, masks)
            batch_recall = recall(outputs, masks)
            batch_f1 = f1_score(outputs, masks)

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

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

        # Log epoch metrics to WandB
        wandb.log({
            "Train Loss": epoch_loss,
            "Train IoU": epoch_iou,
            "Train Dice": epoch_dice,
            "Train Precision": epoch_precision,
            "Train Recall": epoch_recall,
            "Train F1": epoch_f1
        })

        # Validation after each epoch
        teacher.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits, _ = teacher(images)

                # Convert to probabilities
                outputs = torch.sigmoid(logits)

                # Calculate validation loss
                bce_loss = bce_criterion(outputs, masks)
                dice_loss = criterion_dice(outputs, masks)

                loss = 0.5 * bce_loss + 0.5 * dice_loss 

                val_loss += loss.item()

                # Calculate validation metrics
                val_iou += iou(outputs, masks).item()
                val_dice += dice_coefficient(outputs, masks).item()
                val_precision += precision(outputs, masks).item()
                val_recall += recall(outputs, masks).item()
                val_f1 += f1_score(outputs, masks).item()

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

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                # Obtén un lote de validación y pasa las imágenes por el modelo
                sample_images, sample_masks = next(iter(val_loader))
                sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)
                sample_outputs, _ = teacher(sample_images)
                sample_outputs = (torch.sigmoid(sample_outputs) > 0.5).float()  # Binarizar predicciones

                # Seleccionar una imagen aleatoria del lote
                random_index = random.randint(0, sample_images.size(0) - 1)
                
                wandb.log({
                            "Validation Example": wandb.Image(
                                sample_images[random_index].cpu(), 
                                caption="Image Example WHU",
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
        teacher.train()

        if epoch == 0:
            min_val_loss = val_loss
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(teacher.state_dict(), "checkpoints/teacher_alabama_resunet_best_model.pth")
        
if __name__ == "__main__":
    typer.run(train_teacher)

# To run the training script as a command line application in the background, use the following command:
# nohup python train_teacher.py config.yaml > logs/teacher_training.log 2>&1 &
# nohup python train_student.py config.yaml > logs/student_training.log 2>&1 &

## Cambiar los pesos de perdida
## Modificar las capas (agregar capas de convolución, atención)
## Modificar el optimizador y tasa de aprendizaje (callback de reducción de tasa de aprendizaje)
## WHU, LEVIR y CD (3 datasets): https://github.com/likyoo/open-cd
## Añadir imagenes al log de wandb

