import torch
import torch.nn as nn
#from torchvision import transforms
from torchvision.transforms import v2
from transformers import ViTModel

class ViTPatchFeature(nn.Module):
    def __init__(self, unfreeze_last_n=12):
        super().__init__()
        
        # self.preprocess = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.Grayscale(num_output_channels=3), 
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # ])
        self.transforms = v2.Compose([
            #v2.Resize((224, 224), antialias=True),
            v2.Normalize(mean=[0.5], std=[0.5]) # Normalize grayscale first
        ])
        
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # Freeze all, then selectively unfreeze
        for param in self.vit.parameters():
            param.requires_grad = False

        total_layers = len(self.vit.encoder.layer)
        for i in range(total_layers - unfreeze_last_n, total_layers):
            for param in self.vit.encoder.layer[i].parameters():
                param.requires_grad = True
        for param in self.vit.layernorm.parameters():
            param.requires_grad = True
    
    def forward(self, pixel_values):
        # Expect pixel_values to be preprocessed and of shape [B, 3, 224, 224]
        outputs = self.vit(pixel_values=pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]
        # Reshape to spatial grid: [B, 196, 768] → [B, 14, 14, 768]
        B, N, C = patch_tokens.shape
        grid_size = int(N**0.5)  # Assuming N is a perfect square
        patch_map = patch_tokens.reshape(B, grid_size, grid_size, C)
        patch_map = patch_map.permute(0, 3, 1, 2)  # [B, C, 14, 14]
        return patch_map
    

