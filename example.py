import torch
from ast_torch.model import ASTransformer

x = torch.randn(2, 16, 4, 4)
model = ASTransformer(dim=16, dim_head=4, heads=4, depth=1)
attended = model(x)
print(attended)
