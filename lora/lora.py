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
        x = x.view(-1,768)
        x = (self.alpha/self.r) * x @ self.A.T @ self.B.T
        return x.view(x_shape)

def make_hook(i, LoRAs):
    def hook_fn(module, input, output):
        output[:,:,:768] = output[:,:,:768] + LoRAs[i][0](input[0])
        output[:,:,1536:] = output[:,:,1536:] + LoRAs[i][1](input[0])
        return output
    return hook_fn
