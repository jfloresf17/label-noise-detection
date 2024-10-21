import torch
import torchmetrics
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

## Define the teacher model
class TeacherUNetModel(pl.LightningModule):
    def __init__(self, config):
        super(TeacherUNetModel, self).__init__()
        
        # Using a U-Net from segmentation_models_pytorch (smp)
        self.model = smp.Unet(
            encoder_name= config["teacher_model"]["encoder_name"],        # Encoder backbone
            encoder_weights=config["teacher_model"]["encoder_weights"],   # Pre-trained weights for the encoder
            in_channels=config["teacher_model"]["in_channels"],        # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
            classes=config["teacher_model"]["out_channels"]           # Number of output classes for segmentation
        )
        ## Define learning rate
        self.lr = float(config["teacher_model"]["learning_rate"])

        ## Define loss function
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        ## Define metrics   
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
        images, labels = batch
        outputs = self(images)

        # Update the metrics
        ce_loss = self.loss(outputs, labels)

        # Log metrics
        precision = self.train_precision(outputs, labels)
        recall = self.train_recall(outputs, labels)
        iou = self.train_iou(outputs, labels)
        f1 = self.train_f1(outputs, labels)
        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_iou', iou)
        self.log('train_f1', f1)
        self.log('train_loss', ce_loss)
        return {'loss': ce_loss}
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # Update the metrics
        ce_loss = self.loss(outputs, labels)

        # Log metrics
        precision = self.val_precision(outputs, labels)
        recall = self.val_recall(outputs, labels)
        iou = self.val_iou(outputs, labels)
        f1 = self.val_f1(outputs, labels)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_iou', iou)
        self.log('val_f1', f1)
        self.log('val_loss', ce_loss)
        return {'loss': ce_loss}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)

        # Update the metrics
        ce_loss = self.loss(outputs, labels)

        # Log metrics
        precision = self.test_precision(outputs, labels)
        recall = self.test_recall(outputs, labels)
        iou = self.test_iou(outputs, labels)
        f1 = self.test_f1(outputs, labels)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_iou', iou)
        self.log('test_f1', f1)
        self.log('test_loss', ce_loss)     
        return {'loss': ce_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer