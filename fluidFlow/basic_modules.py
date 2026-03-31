import torch.nn as nn
import torch.nn.functional as F

# borrowed from LightningDiT https://github.com/hustvl/LightningDiT/blob/main/models/swiglu_ffn.py
class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        bias=True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    # @torch.compile
    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)