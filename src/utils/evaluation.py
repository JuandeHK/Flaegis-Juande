import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

class DetectorEvaluator:
    """Evalúa el rendimiento del detector de clientes maliciosos."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
    
    def calculate_accuracy(self, total_clients: List[int], 
                         malicious_clients: List[int], 
                         detected_clients: List[int]) -> float:
        """
        Calcula la precisión del detector.
        
        Args:
            total_clients: Lista de todos los clientes
            malicious_clients: Lista de clientes realmente maliciosos
            detected_clients: Lista de clientes detectados como maliciosos
        """
        total_clients = sorted(total_clients)
        malicious_clients = sorted(malicious_clients)
        detected_clients = sorted(detected_clients)
        
        if len(malicious_clients) > 0:
            y_true = [1 if num in malicious_clients else 0 for num in total_clients]
        else:
            y_true = [0 for _ in total_clients]
            
        y_pred = [1 if num in detected_clients else 0 for num in total_clients]
        
        accuracy = accuracy_score(y_true, y_pred)
        self._save_accuracy(accuracy)
        return accuracy
    
    def _save_accuracy(self, accuracy: float):
        """Guarda la precisión en un archivo CSV."""
        path = self.output_dir / "detector_accuracy.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Accuracy"])
        new_df = pd.DataFrame([accuracy], columns=["Accuracy"])
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(path, index=False) 