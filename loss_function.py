import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ImprovedMAELossBinary(nn.Module):
    def __init__(self, T=8):
        """
        Improved Mean Absolute Error Loss (iMAE) para segmentación binaria.
        :param T: Hiperparámetro para ajustar la ponderación, usualmente 8 para etiquetas ruidosas.
        """
        super(ImprovedMAELossBinary, self).__init__()
        self.T = T

    def forward(self, input, target):
        """
        Calcula la pérdida iMAE para segmentación binaria.
        :param input: Tensor de predicciones sin normalizar (logits), de forma (batch_size, 1, H, W).
        :param target: Tensor de valores verdaderos (máscara binaria), de forma (batch_size, 1, H, W).
        :return: Escalar con el valor de la pérdida iMAE.
        """
        # Aplicar la función sigmoide para convertir los logits en probabilidades
        probs = torch.sigmoid(input)
        
        # Calcular los pesos no lineales propuestos
        weights = torch.exp(self.T * probs) * (1 - probs)
        
        # Calcular el error absoluto
        error = torch.abs(probs - target)
        
        # Ponderar el error absoluto
        weighted_error = weights * error
        
        # Promediar todos los errores ponderados
        loss = torch.mean(weighted_error)
        
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred, treshold=0.5):
        y_pred = (y_pred > treshold).float()
        
        ## Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = 1 -((2 * intersection + self.smooth) / (union + self.smooth))
        return dice


    def to(self, device):
        super(DiceLoss, self).to(device)
        return self


def LovaszLoss(logits, labels):
    """
    Binary Lovasz hinge loss
        logits: [B, C, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """

    if len(labels) == 0:
        return logits.sum() * 0.
    
    ## Flatten the logits and labels
    logits = logits.view(-1)
    labels = labels.view(-1)

    ## Compute the loss
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    
    p = len(gt_sorted)

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

    loss = torch.dot(F.relu(errors_sorted), Variable(jaccard))

    return loss

def LovaszLossBatch(logits, labels):
    """
    Lovasz Loss aplicado a un batch de imágenes.
        logits: [B, C, H, W] Variable, logits at cada pixel
        labels: [B, H, W] Tensor, etiquetas binarias de ground truth (0 o 1)
    """
    batch_loss = 0.
    for i in range(logits.size(0)):  # Iterar sobre cada imagen en el batch
        loss = LovaszLoss(logits[i], labels[i])
        batch_loss += loss
    
    # Promedio de pérdidas sobre el batch
    batch_loss = batch_loss / logits.size(0) # batch_loss /= logits.size(0)
    return batch_loss


class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        # BCE
        bce = self.bce_loss(pred, labels)

        # RCE
        pred = torch.sigmoid(pred)  # Apply sigmoid activation for binary segmentation
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_clamped = torch.clamp(labels, min=1e-4, max=1.0)
        rce = (-1 * (label_clamped * torch.log(pred) + (1 - label_clamped) * torch.log(1 - pred)))

        # Loss
        loss = self.alpha * bce + self.beta * rce.mean()
        return loss
    

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, q):
        """
        Implementación de Generalized Cross Entropy (GCE) para segmentación binaria.
        Basado en el paper: "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels"
        """
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Tensor de predicciones (logits), sin aplicar activación.
            y_true: Tensor de etiquetas binarias (0 o 1), de la misma forma que y_pred.
        """
        # Aplicar la activación sigmoide a las predicciones
        y_pred = torch.sigmoid(y_pred)

        # Evitar divisiones por cero y valores fuera del rango válido
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0)

        # Calcular el término intermedio según la fórmula
        intermed = torch.pow(torch.abs(y_true * y_pred + (1 - y_true) * (1 - y_pred)), self.q)
        loss = (1 - intermed) / self.q

        # Promediar la pérdida
        return torch.mean(loss)
