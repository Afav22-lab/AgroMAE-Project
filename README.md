# 🌾 AgroMAE: Modelado Auto-Supervisado de Imágenes Satelitales Multiespectrales para Uso de Suelos

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Afav22-lab/AgroMAE-Project/blob/main/AgroMAE_Entrenamiento_Colab.ipynb)

**Asignatura:** Inteligencia Artificial II — Universidad Industrial de Santander (UIS)  
**Integrantes:**
* Andrés Felipe Alfonso Viviescas
* Andrés Felipe Higuera Grateron

---

## 📌 Descripción del Proyecto

**AgroMAE** implementa un **Masked Autoencoder (MAE)** basado en *Vision Transformers (ViT)* diseñado para procesar de forma nativa las **13 bandas espectrales** de la misión satelital **Sentinel-2** (incluyendo el Infrarrojo Cercano `NIR B08`, `Red-Edge` y `SWIR`).

El proyecto se divide en dos fases:
1. **Fase 1 (Pre-entrenamiento Auto-Supervisado):** Se divide cada imagen satelital ($64 \times 64 \times 13$) en $64$ parches de $8 \times 8$ y se oculta aleatoriamente el **75%** de ellos. El *Encoder* procesa únicamente el 25% visible ($16$ parches) y el *Decoder* ligero reconstruye las 13 bandas faltantes minimizando la pérdida **MSE** sobre los parches enmascarados.
2. **Fase 2 (Transferencia / Clasificación con Escasez de Etiquetas):** Se descarta el *Decoder* y se acopla un cabezal clasificador al *Encoder* pre-entrenado para clasificar **10 coberturas de uso de suelo (EuroSAT)** utilizando únicamente el **10% de los datos etiquetados**, comparando contra un entrenamiento supervisado desde cero (*Scratch*).

---

## 🗂️ Estructura del Repositorio

```text
AgroMAE/
├── AgroMAE_Entrenamiento_Colab.ipynb   # Cuaderno principal (Pre-entrenamiento MAE + Checkpointing + Fine-Tuning)
├── docs/
│   └── entrega2_estructura_y_guion.md  # Estructura y guion cronometrado de sustentación (2da Entrega)
├── requirements.txt                    # Dependencias de Python / PyTorch
├── .gitignore                          # Exclusión de pesos pesados (.pth) y datasets (.zip / .tif)
└── README.md                           # Documentación principal del repositorio
```

---

## ⚙️ Características Técnicas Clave

* **Aceleración Hardware (GPU NVIDIA L4 / Ada Lovelace):** Soporte nativo de precisión mixta `bfloat16`/`float16` (`torch.amp`), `TF32` y *FlashAttention* (`F.scaled_dot_product_attention`).
* **Gestor de Checkpoints de Doble Nivel (`CheckpointManager`):**
  * **Nivel Intra-Epoch:** Guardado atómico cada 200 pasos en Google Drive con bandera `epoch_completed` para reanudar sin pérdida de progreso ante desconexiones de Google Colab.
  * **Nivel Epoch:** Registro automático del mejor modelo histórico (`checkpoint_best.pth`) incluyendo pesos, momentos de `AdamW`, `CosineAnnealingLR` y `GradScaler`.
* **Partición Estratificada:** Separación 85% *Train* / 15% *Validation* y submuestreo estratificado del 10% mediante `StratifiedShuffleSplit`.

---

## 🚀 Cómo Ejecutar en Google Colab Directamente desde GitHub

1. Abre [Google Colab](https://colab.research.google.com/).
2. Selecciona la pestaña **GitHub**, pega la URL de este repositorio y abre `AgroMAE_Entrenamiento_Colab.ipynb`.
3. En **Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno de ejecución**, selecciona **GPU L4** (o T4) y activa **RAM amplia**.
4. Ejecuta todas las celdas (`Ctrl + F9`).

---

## 📚 Referencias

1. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022).** *Masked Autoencoders Are Scalable Vision Learners.* IEEE/CVF CVPR, 16000–16009.
2. **Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).** *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 12(7), 2217–2226.
3. **Cong, Y., et al. (2022).** *SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.* NeurIPS.

