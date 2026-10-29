# 🛰️ AgroMAE: Modelado Auto-Supervisado de Imágenes Satelitales para Uso de Suelos

**Proyecto de Inteligencia Artificial II**  
**Integrantes:**
- Andrés Felipe Alfonso Viviescas
- Andrés Felipe Higuera Grateron

---

## 📌 1. Descripción del Proyecto

El monitoreo agrícola y de deforestación a gran escala exige modelos de visión artificial capaces de procesar imágenes satelitales multiespectrales (Sentinel-2, 13 bandas). Sin embargo, la adquisición de anotaciones y etiquetas manuales a nivel de píxel es costosa e inviable.

**AgroMAE** aborda este desafío implementando un **Masked Autoencoder (MAE)** basado en Vision Transformers (ViT) que:
1. Oculta agresivamente el **75% de los parches** de una imagen multiespectral ($64\times 64\times 13$).
2. Aprende de manera **auto-supervisada (sin etiquetas)** a reconstruir tanto la geometría espacial como la respuesta espectral y vegetal (infrarrojo cercano / NIR).
3. Permite transferir el conocimiento del *Encoder* para clasificar el uso del suelo con alta precisión utilizando **únicamente el 10% de los datos etiquetados**, superando a las arquitecturas convolucionales clásicas (CNNs) entrenadas desde cero.

---

## 👥 2. Organización del Equipo y Flujo de Trabajo

Para maximizar la eficiencia y garantizar un trabajo **50/50 equitativo**:

| Colaborador | Rol Principal | Responsabilidades |
| :--- | :--- | :--- |
| **Andrés Felipe Alfonso** | *GPU Lead & Autoencoder* | • Pre-entrenamiento de AgroMAE (cómputo pesado en GPU/Colab)<br>• CheckpointManager y persistencia en Google Drive<br>• Fine-Tuning downstream del clasificador AgroMAE<br>• Generación de mosaicos de reconstrucción multiespectral (RGB y NIR) |
| **Andrés Felipe Higuera** | *Benchmark & Evaluación* | • Implementación y entrenamiento del Modelo A (Baseline CNN supervisada)<br>• Pipeline unificado de métricas (F1-Score Macro, Accuracy, Matriz de Confusión)<br>• Análisis comparativo de curvas de desempeño (% etiquetas vs precisión)<br>• Diapositivas e informe técnico |

> **Nota sobre Dependencias:**  
> Ambos integrantes pueden trabajar en paralelo sin bloquearse gracias al módulo `src/data/splits.py`, que fija una semilla determinista (`seed=42`) para garantizar que ambos modelos se evalúen sobre el mismo conjunto de validación y prueba.

---

## 📁 3. Estructura del Repositorio

```text
AgroMAE-Project/
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── data/
│   │   ├── dataset.py            # Carga multiespectral Sentinel-2 (13 bandas) y normalización física (/10000)
│   │   └── splits.py              # Particiones deterministas y extracción de subsets (ej. 10% etiquetado)
│   ├── models/
│   │   ├── agromae.py             # Arquitectura Masked Autoencoder (ViT + Sincos 2D + Decoder ligero)
│   │   ├── classifier.py          # Cabezal downstream sobre el Encoder pre-entrenado
│   │   └── baseline_cnn.py        # Modelo A: CNN convolucional adaptada a 13 bandas
│   └── utils/
│       ├── checkpoint.py          # CheckpointManager tolerante a desconexiones de Colab
│       ├── metrics.py             # Cálculo de Accuracy, Macro F1 y Matrices de Confusión
│       └── plotting.py            # Generación de curvas de pérdida y mosaicos de 4 paneles en alta resolución
└── notebooks/
    └── 01_AgroMAE_Preentrenamiento.ipynb  # Cuaderno listo para Google Colab con aceleración GPU L4/T4
```

---

## 🚀 4. Guía de Ejecución Rápida

### Instalación de Dependencias
```bash
pip install -r requirements.txt
pip install rasterio
```

### Dataset (EuroSAT Multiespectral)
El dataset oficial de EuroSAT (Sentinel-2, 13 bandas, 10 clases) se descarga directamente desde Zenodo:
```bash
wget https://zenodo.org/records/7745585/files/EuroSATallBands.zip
unzip -q EuroSATallBands.zip -d eurosat_ms/
```

---

## 📊 5. Rúbrica de la Segunda Entrega (Presentación Oral 10 min)

- **Portada** (15 seg)
- **Breve Introducción** (1 min)
- **Objetivo / Alcance** (45 seg)
- **Método Propuesto** (3 min)
- **Configuración Experimental** (1 min)
- **Resultados Preliminares** (2 min)
- **Conclusiones y Trabajo Faltante** (1 min)
- **Vistazo al Notebook** (1 min)
