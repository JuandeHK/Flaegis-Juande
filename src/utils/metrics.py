from typing import List, Dict
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

class MetricsCalculator:
    """Calcula métricas de evaluación del modelo."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula todas las métricas relevantes.
        
        Returns:
            Dict con todas las métricas calculadas
        """
        return {
            "accuracy": np.mean(y_true == y_pred),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
            "fpr": MetricsCalculator._calculate_weighted_fpr(y_true, y_pred)
        }
    
    @staticmethod
    def _calculate_weighted_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula FPR ponderado."""
        cm = confusion_matrix(y_true, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (fp + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
        
        class_weights = np.sum(cm, axis=1) / np.sum(cm)
        fpr_per_class = fp / (fp + tn)
        
        return np.sum(fpr_per_class * class_weights) 