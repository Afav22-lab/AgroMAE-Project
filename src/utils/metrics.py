import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred):
    """
    Calcula métricas clave para la evaluación del uso de suelos:
    - Overall Accuracy
    - Macro F1-Score
    - Matriz de confusión
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
        
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm
    }
