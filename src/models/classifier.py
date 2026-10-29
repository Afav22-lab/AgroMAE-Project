import torch
import torch.nn as nn

class AgroMAEClassifier(nn.Module):
    """
    Clasificador Downstream para uso de suelos que reutiliza el Encoder de AgroMAE.
    """
    def __init__(self, pretrained_mae, num_classes=10, freeze_encoder=False):
        super().__init__()
        self.patch_embed = pretrained_mae.patch_embed
        self.pos_embed = pretrained_mae.pos_embed
        self.blocks = pretrained_mae.blocks
        self.norm = pretrained_mae.norm
        
        if freeze_encoder:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for p in self.blocks.parameters():
                p.requires_grad = False
            for p in self.norm.parameters():
                p.requires_grad = False
                
        embed_dim = pretrained_mae.pos_embed.shape[-1]
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # En clasificación, la imagen entra 100% visible (sin máscara)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        # Global Average Pooling espacial
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits
