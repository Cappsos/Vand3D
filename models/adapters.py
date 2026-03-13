from torch import Tensor, nn
import torch
from torch.nn import functional as F

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model_type_str):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
        self.model_type_str = model_type_str.lower()

    def forward(self, tokens_list):
        if not isinstance(tokens_list, list):
            raise TypeError("Input to LinearLayer.forward must be a list of tensors.")
        
        output_tokens_list = []
        for i in range(len(tokens_list)):
            current_tokens = tokens_list[i] # Shape: (B, NumPatches, HiddenDim) for ViT

            if 'vit' in self.model_type_str: # Check if it's ViT features
                if current_tokens.ndim != 3:
                    raise ValueError(f"Expected 3D tensor (B, NumPatches, HiddenDim) for ViT features, got {current_tokens.shape}")
                adapted_tokens = self.fc[i](current_tokens)
                output_tokens_list.append(adapted_tokens)
            else:
                if current_tokens.ndim == 3:
                    print(f"Warning: Non-ViT path in LinearLayer activated with shape {current_tokens.shape}. Review logic.")
                    # Fallback or specific ResNet logic:
                    B, C_dim, H_dim, W_dim = current_tokens.shape # Assuming it's B,C,H,W
                    reshaped_for_fc = current_tokens.view(B, C_dim, -1).permute(0, 2, 1).contiguous()
                    adapted_tokens = self.fc[i](reshaped_for_fc) # This assumes fc[i] input dim matches C_dim
                    output_tokens_list.append(adapted_tokens)

                else: # (B, C, H, W) original for ResNet blocks
                    B, C_in_layer, H, W = current_tokens.shape
                    reshaped_tokens = current_tokens.view(B, C_in_layer, -1).permute(0, 2, 1).contiguous()
                    adapted_tokens = self.fc[i](reshaped_tokens)
                    output_tokens_list.append(adapted_tokens)
        return output_tokens_list




