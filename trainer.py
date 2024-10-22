import torch
import pathlib
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.student_unet import StudentUNetModel
from models.teacher_unet import TeacherUNetModel
from dataloader import WHUDataModule, DataCentricDataModule
import typer
from utils import load_yaml

# Check if GPU is available, and if not, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Teacher Model
def train_teacher(config):
    # Initialize teacher model
    teacher_model = TeacherUNetModel(config)
    
    ## Num workers y batch size
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]

    # Paths a las imágenes y etiquetas
    CLEAN_DATASET = pathlib.Path(config["data"]["whu_path"])
    
    ## Mean and std from WHU dataset
    if config["Normalize"]["apply"]:
        mean_whu, std_whu = config["Normalize"]["WHU"]["mean"], config["Normalize"]["WHU"]["std"]

        ## Initialize the dataloader
        clean_dataloader = WHUDataModule(CLEAN_DATASET, normalize=True, mean=mean_whu, std=std_whu, 
                                         batch_size=batch_size, num_workers=num_workers)
    
    else:
        clean_dataloader = WHUDataModule(CLEAN_DATASET, normalize=False, mean=None, std=None,
                                         batch_size=batch_size, num_workers=num_workers) 
        
    clean_dataloader.setup()

    # Initialize the callbacks
    early_stop_params = config["trainer"]["early_stopping"]
    early_stopping = pl.callbacks.EarlyStopping(        
        monitor=early_stop_params["monitor"],
        patience=early_stop_params["patience"],
        mode=early_stop_params["mode"]
    )
   
    checkpoint_params = config["trainer"]["checkpoint_callback"]
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=checkpoint_params["monitor"],        
        dirpath=config["teacher_model"]["checkpoint_dir"],
        filename=config["teacher_model"]["checkpoint_name"],
        mode=checkpoint_params["mode"],
        save_top_k=checkpoint_params["save_top_k"]
    )

    callbacks = [early_stopping, checkpoint]

    # Initialize the logger
    project_name = config['trainer']['wandb_project']
    experiment_name = config["trainer"]["experiment_name"]
    teacher_wandb_logger = WandbLogger(project=project_name,
                                       name=experiment_name)

    # Initialize the trainer
    trainer = pl.Trainer(
            strategy=config["trainer"]["strategy"],
            accelerator=config["trainer"]["accelerator"],
            devices=torch.cuda.device_count(),
            max_epochs=config["trainer"]["max_epochs"],
            callbacks=callbacks,
            precision=config["trainer"]["precision"],
            log_every_n_steps=config["trainer"]["log_every_n_steps"],
            logger=teacher_wandb_logger
        )
    
    
    # Train teacher model
    trainer.fit(teacher_model, clean_dataloader)
    
    # Test teacher model
    trainer.test(teacher_model, clean_dataloader, ckpt_path="best")


# Training Student Model
def train_student(config):
    # Initialize student model
    student_model = StudentUNetModel(config)   
    
    ## Num workers y batch size
    batch_size = config["data"]["batch_size"]
    num_workers = config["data"]["num_workers"]
  
    INPUT_FOLDER = pathlib.Path(config["data"]["datacentric_image_path"])
    LABEL_FOLDER = pathlib.Path(config["data"]["label_noisy_path"])

    ## Real teacher outputs
    TEACHER_FOLDER = pathlib.Path(config["data"]["teacher_output_path"])

    ## Get the files
    input_files = sorted(list(INPUT_FOLDER.glob('*.png')))
    noisy_label_files = sorted(list(LABEL_FOLDER.glob('*.png')))
    teacher_output_files = sorted(list(TEACHER_FOLDER.glob('*.png'))) # Real teacher outputs

    ## From Data-centric dataset
    if config["Normalize"]["apply"]:
        mean_dc, std_dc = config["Normalize"]["DataCentric"]["mean"], config["Normalize"]["DataCentric"]["std"]

        # Initialize the datasets
        student_dataloader = DataCentricDataModule(input_files, noisy_label_files, teacher_output_files,
                                                   normalize=True, mean=mean_dc, std=std_dc, 
                                                   batch_size=batch_size, num_workers=num_workers)
        
    else:
        student_dataloader = DataCentricDataModule(input_files, noisy_label_files, teacher_output_files,
                                                   normalize=False, mean=None, std=None, 
                                                   batch_size=batch_size, num_workers=num_workers)
    student_dataloader.setup()

    # Initialize the callbacks
    early_stop_params = config["trainer"]["early_stopping"]
    early_stopping = pl.callbacks.EarlyStopping(        
        monitor=early_stop_params["monitor"],
        patience=early_stop_params["patience"],
        mode=early_stop_params["mode"]
    )
   
    checkpoint_params = config["trainer"]["checkpoint_callback"]
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=checkpoint_params["monitor"],
        dirpath=config["student_model"]["checkpoint_dir"],
        filename=config["student_model"]["checkpoint_name"],
        mode=checkpoint_params["mode"],
        save_top_k=checkpoint_params["save_top_k"]
    )

    callbacks = [early_stopping, checkpoint]

    # Initialize the logger
    project_name = config['trainer']['wandb_project']
    experiment_name = config["trainer"]["experiment_name"]
    teacher_wandb_logger = WandbLogger(project=project_name, 
                                       name=experiment_name)

    # Initialize the trainer
    trainer = pl.Trainer(
            strategy=config["trainer"]["strategy"],
            accelerator=config["trainer"]["accelerator"],
            devices=torch.cuda.device_count(),
            max_epochs=config["trainer"]["max_epochs"],
            callbacks=callbacks,
            precision=config["trainer"]["precision"],
            log_every_n_steps=config["trainer"]["log_every_n_steps"],
            logger=teacher_wandb_logger
        )
    
    
    # Train teacher model
    trainer.fit(student_model, student_dataloader)
    
    # Test teacher model
    trainer.test(student_model, student_dataloader, ckpt_path="best")


app = typer.Typer()

@app.command("train")
def train(model: str = typer.Option(..., "--model"), 
          config_path: str = typer.Option(..., "--config_path")):
    
    ## Load the configuration file
    config = load_yaml(config_path)

    ## Train the model
    if model == "teacher":
        train_teacher(config)
    elif model == "student":
        train_student(config)
    else:
        typer.echo("Invalid model type. Please use 'teacher' or 'student'.")

if __name__ == "__main__":
    app()

## Run in terminal
## nohup python trainer.py --model=teacher --config_path=config.yaml > logs/teacher.log 2>&1 &
## nohup python trainer.py --model=student --config_path=config.yaml > logs/student.log 2>&1 &