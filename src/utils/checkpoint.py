import os
import torch

class CheckpointManager:
    """
    Gestor de checkpoints tolerante a fallos y desconexiones de Google Colab.
    """
    def __init__(self, checkpoint_dir, max_to_keep=2):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.last_path = os.path.join(checkpoint_dir, "checkpoint_last.pth")
        self.best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")

    def save(self, epoch, global_step, model, optimizer, scheduler=None, scaler=None, loss=None, is_best=False):
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': loss
        }
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)
            print(f"⭐ [RECORD] Mejor modelo guardado con métrica: {loss:.4f}")

    def load_latest(self, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
        if not os.path.exists(self.last_path):
            print("🚀 No se encontró checkpoint previo. Iniciando desde cero.")
            return 0, 0, float("inf")

        print(f"🔄 Reanudando desde checkpoint: {self.last_path}")
        ckpt = torch.load(self.last_path, map_location=device)
        
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and ckpt.get('optimizer_state_dict'):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and ckpt.get('scheduler_state_dict'):
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and ckpt.get('scaler_state_dict'):
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        start_epoch = ckpt.get('epoch', 0)
        start_step = ckpt.get('global_step', 0)
        best_loss = ckpt.get('loss', float("inf"))

        print(f"✅ Estado reanudado en época {start_epoch + 1}, paso {start_step}.")
        return start_epoch + 1, start_step, best_loss
