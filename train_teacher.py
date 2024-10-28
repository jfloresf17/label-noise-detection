import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import typer

# Import the UNetTeacher and WHUDataModule classes
from utils import load_config
from models.unet_model import UNetTeacher
from dataloader import WHUDataModule
from scores import iou, dice_coefficient, precision, recall, f1_score

def train_teacher(config_path: str):
    # Load configuration from YAML file
    config = load_config(config_path)

    file_paths = config["data"]["whu_path"]
    normalize = config['Normalize']["apply"]
    mean = config['Normalize']["WHU"]['mean']
    std = config['Normalize']["WHU"]['std']
    batch_size = config["data"]['batch_size']
    num_workers = config["data"]['num_workers']
    lr = float(config['learning_rate'])
    epochs = config["trainer"]['max_epochs']
    device = config["device"]

    # Initialize Weights & Biases
    wandb.init(project=config["trainer"]["wandb_project"], config=config, 
               name=config["trainer"]["experiment_name"])

    # Define the Teacher model
    teacher = UNetTeacher(n_channels=3, n_classes=1)
    teacher = teacher.to(device)  # Use GPU if available

    # Define the optimizer and loss function
    optimizer = optim.Adam(teacher.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss().to(device)  # For binary segmentation, we use Binary Cross Entropy with Logits
    scheduler = ReduceLROnPlateau(optimizer, # Reduce learning rate when a val loss metric has stopped improving 
                                  mode='min', # Minimize the loss
                                  factor=0.5, # Reduce the learning rate by half 
                                  patience=3 # Number of epochs with no improvement after which learning rate will be reduced 
                                  )

    # Create the data module
    data_module = WHUDataModule(file_paths, normalize, mean, std, batch_size=batch_size, num_workers=num_workers)
    data_module.setup(stage='fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

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
            outputs, _ = teacher(images)
            loss = criterion(outputs, masks)

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
                outputs, _ = teacher(images)
                loss = criterion(outputs, masks)

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
                sample_outputs = torch.sigmoid(sample_outputs)  # Convertir logits a probabilidades
                sample_outputs = (sample_outputs > 0.5).float()  # Binarizar predicciones

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
            torch.save(teacher.state_dict(), "checkpoints/teacher_best_model.pth")
        
if __name__ == "__main__":
    typer.run(train_teacher)

# To run the training script as a command line application in the background, use the following command:
# nohup python train_teacher.py config.yaml > logs/teacher_training.log 2>&1 &

## Cambiar los pesos de perdida
## Modificar las capas (agregar capas de convolución, atención)
## Modificar el optimizador y tasa de aprendizaje (callback de reducción de tasa de aprendizaje)
## WHU, LEVIR y CD (3 datasets): https://github.com/likyoo/open-cd
## Añadir imagenes al log de wandb

