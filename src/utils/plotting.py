import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_loss_curves(train_losses, val_losses, output_path="loss_curves.png"):
    """
    Genera y guarda la curva de convergencia de pérdida para las diapositivas.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, train_losses, label="Pérdida Entrenamiento (Train Loss)", color="#2563eb", lw=2)
    plt.plot(epochs, val_losses, label="Pérdida Validación (Val Loss)", color="#16a34a", lw=2, linestyle="--")
    plt.title("Convergencia del Pre-entrenamiento Auto-Supervisado (AgroMAE)", fontsize=13, fontweight="bold")
    plt.xlabel("Épocas", fontsize=11)
    plt.ylabel("MSE de Reconstrucción Espectral", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Gráfica de pérdidas guardada en: {output_path}")

def plot_and_save_reconstruction_mosaic(model, dataloader, device, output_path="reconstruction_mosaic.png", num_samples=3):
    """
    Genera y guarda en alta resolución el mosaico comparativo de 4 paneles para las diapositivas.
    Columnas:
    1. Original RGB (B04, B03, B02)
    2. Entrada Enmascarada al 75%
    3. Reconstrucción AgroMAE
    4. Falso Color Infrarrojo de Vegetación NIR (B08, B04, B03)
    """
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:num_samples].to(device)

    with torch.no_grad():
        _, pred, mask = model(imgs)
        pred_imgs = model.unpatchify(pred)

        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, model.patch_size ** 2 * model.in_chans)
        mask_imgs = model.unpatchify(mask_expanded)
        masked_inputs = imgs * (1.0 - mask_imgs)

    imgs = imgs.cpu().float().numpy()
    masked_inputs = masked_inputs.cpu().float().numpy()
    pred_imgs = pred_imgs.cpu().float().numpy()

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples), dpi=300)
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        # 1. RGB Original (B04, B03, B02)
        orig_rgb = np.clip(np.stack([imgs[i, 3], imgs[i, 2], imgs[i, 1]], axis=-1) * 3.5, 0, 1)
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title("Original (RGB Sentinel-2)", fontsize=11, fontweight="bold")
        axes[i, 0].axis('off')

        # 2. Entrada 75% Oculta
        masked_rgb = np.clip(np.stack([masked_inputs[i, 3], masked_inputs[i, 2], masked_inputs[i, 1]], axis=-1) * 3.5, 0, 1)
        axes[i, 1].imshow(masked_rgb)
        axes[i, 1].set_title("Entrada Enmascarada (75% Oculto)", fontsize=11, fontweight="bold")
        axes[i, 1].axis('off')

        # 3. Reconstrucción AgroMAE
        pred_rgb = np.clip(np.stack([pred_imgs[i, 3], pred_imgs[i, 2], pred_imgs[i, 1]], axis=-1) * 3.5, 0, 1)
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title("Reconstrucción AgroMAE", fontsize=11, fontweight="bold")
        axes[i, 2].axis('off')

        # 4. Falso Color Infrarrojo NIR (B08, B04, B03)
        pred_nir = np.clip(np.stack([pred_imgs[i, 7], pred_imgs[i, 3], pred_imgs[i, 2]], axis=-1) * 3.0, 0, 1)
        axes[i, 3].imshow(pred_nir)
        axes[i, 3].set_title("Falso Color Infrarrojo (NIR Vegetación)", fontsize=11, fontweight="bold")
        axes[i, 3].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"🖼️ Mosaico espectral guardado en: {output_path}")
