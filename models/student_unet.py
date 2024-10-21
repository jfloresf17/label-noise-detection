import torch
import torchmetrics
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class StudentUNetModel(pl.LightningModule):
    def __init__(self, config):
        super(StudentUNetModel, self).__init__()
        
        # Using a U-Net from segmentation_models_pytorch (smp)
        self.model = smp.Unet(
            encoder_name= config["student_model"]["encoder_name"],        # Encoder backbone
            encoder_weights=config["student_model"].get("encoder_weights", None),   # Pre-trained weights for the encoder
            in_channels=config["student_model"]["in_channels"],        # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            classes=config["student_model"]["out_channels"]           # Number of output classes for segmentation
        )
        # Define learning rate
        self.lr = float(config["student_model"]["learning_rate"])

        # Define loss function
        self.loss = torch.nn.BCEWithLogitsLoss()

        ## Define parameters for knowledge distillation
        self.alpha = config["knowledge_distillation"]["alpha"]
        self.beta = config["knowledge_distillation"]["beta"]
        self.temperature = config["knowledge_distillation"]["temperature"]

        
        # Define metrics
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.train_iou = torchmetrics.JaccardIndex(task="binary")
        self.train_precision = torchmetrics.Precision(task="binary")
        self.train_recall = torchmetrics.Recall(task="binary")

        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.val_iou = torchmetrics.JaccardIndex(task="binary")
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_iou = torchmetrics.JaccardIndex(task="binary")
        self.test_precision = torchmetrics.Precision(task="binary")
        self.test_recall = torchmetrics.Recall(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, teacher_outputs = batch
        outputs = self(images)

        # Calculate loss
        supervised_loss = self.loss(outputs, labels)
        
        # Knowledge distillation loss (difference between outputs and teacher outputs)
        distillation_loss = torch.nn.functional.kl_div(
            torch.log_softmax(outputs, dim=1),
            torch.softmax(teacher_outputs, dim=1),
            reduction='batchmean'
        )* (self.temperature ** 2)

        # Combine losses
        total_loss = self.alpha * distillation_loss + self.beta * supervised_loss
        
        # Calculate noise estimation (difference between labels and teacher outputs)
        noise = torch.nn.functional.mse_loss(outputs, teacher_outputs)

       
        # Update metrics
        precision = self.train_precision(outputs, labels)
        recall = self.train_recall(outputs, labels)
        iou = self.train_iou(outputs, labels)
        f1 = self.train_f1(outputs, labels)

        # Log metrics
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_iou', iou)
        self.log('train_f1', f1)
        self.log('train_loss', total_loss)
        self.log('train_noise', noise)
        return {'loss': total_loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels, teacher_outputs = batch
        outputs = self(images)

        # Calculate loss
        supervised_loss = self.loss(outputs, labels)
        
        # Knowledge distillation loss (difference between outputs and teacher outputs)
        distillation_loss = torch.nn.functional.kl_div(
            torch.log_softmax(outputs, dim=1),
            torch.softmax(teacher_outputs, dim=1),
            reduction='batchmean'
        )* (self.temperature ** 2)

        # Combine losses
        total_loss = self.alpha * distillation_loss + self.beta * supervised_loss

        
        # Calculate noise estimation (difference between labels and teacher outputs)
        noise = torch.nn.functional.mse_loss(outputs, teacher_outputs)
        
        # Update metrics
        precision = self.val_precision(outputs, labels)
        recall = self.val_recall(outputs, labels)
        iou = self.val_iou(outputs, labels)
        f1 = self.val_f1(outputs, labels)

        # Log metrics
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_iou', iou)
        self.log('val_f1', f1)
        self.log('val_loss', total_loss)
        self.log('val_noise', noise)
        return {'loss': total_loss}
    
    def test_step(self, batch, batch_idx):
        images, labels, teacher_outputs = batch
        outputs = self(images)

        # Calculate loss
        supervised_loss = self.loss(outputs, labels)
        
        # Knowledge distillation loss (difference between outputs and teacher outputs)
        distillation_loss = torch.nn.functional.kl_div(
            torch.log_softmax(outputs, dim=1),
            torch.softmax(teacher_outputs, dim=1),
            reduction='batchmean'
        )* (self.temperature ** 2)

        # Combine losses
        total_loss = self.alpha * distillation_loss + self.beta * supervised_loss
        
        # Calculate noise estimation (difference between labels and teacher outputs)
        noise = torch.nn.functional.mse_loss(outputs, teacher_outputs)
        
        # Update metrics
        precision = self.test_precision(outputs, labels)
        recall = self.test_recall(outputs, labels)
        iou = self.test_iou(outputs, labels)
        f1 = self.test_f1(outputs, labels)

        # Log metrics
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_iou', iou)
        self.log('test_f1', f1)
        self.log('test_loss', total_loss)
        self.log('test_noise', noise)
        return {'loss': total_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
