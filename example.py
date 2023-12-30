import torch
from ast_torch.model import ASTransformer

# Create dummy data
x = torch.randn(2, 16)

# Initialize model
model = ASTransformer(
    dim=4, dim_head=4, heads=4, depth=2, patch_size=4
)

# Run model and print output shape
print(model(x).shape)
