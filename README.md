# FLAegis FL approach
FL approach for enhanced guarding against intrusion and security threads.
![FLAegis Scheme](/img/FLAegisScheme.png)

## 📖 Abstract
Abstract—Federated Learning (FL) has become a powerful
technique for training Machine Learning (ML) models in a
decentralized manner, preserving the privacy of the training
datasets involved. However, the decentralized nature of FL means
that much of the process cannot be validated or supervised. This
reliance on the honesty of participating clients creates an oppor
tunity for malicious third parties, known as Byzantine clients, to
disrupt the process by sending false information. These malicious
clients may engage in poisoning attacks, manipulating either
the dataset or the model parameters to induce misclassification.
In response, this study introduces FLAegis, a novel defensive
framework designed to identify Byzantine clients and fortify FL
systems. Our approach leverages clustering methodologies and
time series transformations to amplify the differences between
benign and malicious models, enabling their classification based
on these distinctions. Furthermore, we incorporate a robust ag
gregation function as a final layer to mitigate the impact of those
byzantine clients which managed to evade prior defenses. We
rigorously evaluated our method against five distinct poisoning
attacks, from basic to sophisticated ones. Notably, our approach
not only optimizes techniques compared to existing methodologies
but also outperforms them across all attack scenarios, exhibiting
nearly 100% effectiveness in most instances.

## 📕 Documentación
Puedes consultar la [documentación completa aquí](docs/documentation.md).

La documentación está organizada en las siguientes secciones:
- Core (main y orchestrator)
- Cliente
- Servidor
- Estrategias de agregación
- Archivo de configuración 

Actualizala con el comando
```bash
make docs
```

## ▶️ Ejecución
`make run -m [numero clientes maliciosos] -c numero de clientes`  

Ejemplo:
```
make run -m 5 -c 50
make run -- -m 5 -c 50 !!!!!!!!!
python main.py -m 5 -c 50
PYTHONPATH=src python main.py -m 2 -c 10
```

## 🐍 Entorno virtual y requirements.txt
1. Crea un entorno virtual:  
`python -m venv .\.venv`
python3 -m venv .venv
source .venv/bin/activate
make run m=5 c=50


 
2. Activa el entorno virtual  
    En windows: 
    `.\.venv\Scripts\activate`  
    En linux/macos: `source .venv/bin/activate` 
3. Descarga las dependencias:  
`pip install -r requirements.txt`

Salir del entono vitual:  
`exit`

## ⚙️ Configuración YAML

El archivo de configuración principal está disponible en [`config/config.yaml`](config/config.yaml). Para obtener una descripción detallada de cada parámetro y cómo ajustarlo, consulta la [documentación de configuración YAML](docs/yaml_documentation.md).

## ⚠️ Posibles problemas y soluciones
Windows te da problemas con los paths largos al instalar las dependencias (sklearn, etc).
```
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
