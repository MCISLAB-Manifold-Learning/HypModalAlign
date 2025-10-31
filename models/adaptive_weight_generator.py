import torch
import torch.nn as nn

class AdaptiveWeightGenerator(nn.MultiheadAttention):
    def __init__(self, d_model, num_heads=1, device=None, dtype=None):
        super().__init__(embed_dim=d_model, num_heads=num_heads, 
                         device=device, dtype=dtype)
        
        self.d_model = d_model
        self.v_start = 2 * d_model
        
        with torch.no_grad():
            eye_matrix = torch.eye(d_model, device=device, dtype=dtype)
            if self.in_proj_weight.shape[0] > 2 * d_model:
                self.in_proj_weight[self.v_start:, :] = eye_matrix
            
            if self.in_proj_bias is not None:
                self.in_proj_bias[self.v_start:].zero_()
        
        with torch.no_grad():
            self.out_proj.weight = nn.Parameter(torch.eye(
                d_model, device=device, dtype=dtype
            ))
            if self.out_proj.bias is not None:
                self.out_proj.bias.data.zero_()
        
        self.out_proj.weight.requires_grad = False
        if self.out_proj.bias is not None:
            self.out_proj.bias.requires_grad = False
    
    def forward(self, query, key, value, **kwargs):
        return super().forward(
            query=query,
            key=key,
            value=value,
            **kwargs
        )