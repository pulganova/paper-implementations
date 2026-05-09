import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, d, k, r, alpha):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(torch.randn(r,k))
        self.B = nn.Parameter(torch.zeros(d,r))

    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1,x.shape[-1])
        x = (self.alpha/self.r) * x @ self.A.T @ self.B.T
        return x.view(x_shape)

def make_hook(i, LoRAs):
    def hook_fn(module, input, output):
        embed_dim = output.shape[-1] // 3
        output[:,:,:embed_dim] = output[:,:,:embed_dim] + LoRAs[i][0](input[0])
        output[:,:,embed_dim*2:] = output[:,:,embed_dim*2:] + LoRAs[i][1](input[0])
        return output
    return hook_fn
