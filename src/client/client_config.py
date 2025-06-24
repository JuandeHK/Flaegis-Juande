from dataclasses import dataclass
from typing import List
import multiprocessing

@dataclass
class ClientConfig:
    def __init__(self, client_id: str, numeric_id: int, is_malicious: bool):
        self._client_id = client_id
        self._numeric_id = numeric_id
        self._is_malicious = is_malicious  # Usar atributo privado
    
    @property
    def client_id(self):
        return self._client_id
    
    @property
    def numeric_id(self):
        return self._numeric_id
    
    @property
    def is_malicious(self):
        return self._is_malicious
    
    @is_malicious.setter  # ← AÑADIR SETTER
    def is_malicious(self, value: bool):
        self._is_malicious = value
