import torch

def iou(pred, target, threshold=0.5):
    """
    The formula to calculate IoU is:
        IoU = (|X & Y|)/ (|X or Y|)
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection ## Union = A + B - Intersection (FOR THEORY OF SETS)
    
    if union == 0:
        return torch.tensor(0.0) # If there is no ground truth, do not include in evaluation
    
    iou = intersection / union
    return iou

def weighted_iou(pred, target, weight_factor=0.1, epsilon=1e-6):
    """
    Calcula el Weighted IoU (WIoU) entre una predicción y un target.
    Se penalizan las diferencias entre pred y target, ponderadas por un factor.

    Parámetros:
    - pred: Tensor de etiquetas ruidosas.
    - target: Tensor de etiquetas reales.
    - threshold: Umbral para convertir las predicciones en binario.
    - weight_factor: Factor de ponderación para penalizar el ruido.
    - epsilon: Constante para evitar divisiones por cero en el caso de unión cero.

    Retorno:
    - wiou: Weighted IoU entre pred y target.
    """
    # Convertimos pred en binario con el umbral y target 
    pred = pred.float()
    target = target.float()
    
    # Calculamos la intersección y la unión
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    # Calculamos la diferencia entre pred y target
    difference = torch.abs(pred - target).sum()
    
    # Aplicamos epsilon en el denominador para evitar división por cero
    iou = intersection / (union + epsilon)
    
    # Aplicamos la penalización por la diferencia
    wiou = iou - (weight_factor * difference / (union + epsilon))    
   
    return wiou

def dice_coefficient(pred, target, threshold=0.5):
    """
    The formula to calculate Dice Coefficient is:
        Dice = (2 * |X & Y|)/ (|X|+ |Y|)
    """
    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    suma = pred.sum() + target.sum()

    if suma == 0:
        return torch.tensor(0.0)
    
    dice = (2. * intersection) / suma
    return dice


def precision(pred, target, threshold=0.5):
    """
    The formula to calculate Precision is:
        Precision = TP / (TP + FP)
    where:
        TP: True Positives
        FP: False Positives
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()

    if TP + FP == 0:
        return torch.tensor(0.0)
    
    precision = TP / (TP + FP)
    return precision


def recall(pred, target, threshold=0.5):
    """
    The formula to calculate Recall is:
        Recall = TP / (TP + FN)
    where:
        TP: True Positives
        FN: False Negatives
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    TP = (pred * target).sum()
    FN = ((1 - pred) * target).sum()

    if TP + FN == 0:
        return torch.tensor(0.0)
    
    recall = TP / (TP + FN)
    return recall


def f1_score(pred, target, threshold=0.5):
    """
    The formula to calculate F1 Score is:
        F1 = 2 * (precision * recall) / (precision + recall)
    where:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    ## Calculate Precision and Recall
    precision_val = precision(pred, target)
    recall_val = recall(pred, target)

    if precision_val + recall_val == 0:
        return torch.tensor(0.0)
    
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val)
    return f1