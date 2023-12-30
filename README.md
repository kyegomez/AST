[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# AST
Implementation of AST from the paper: "AST: Audio Spectrogram Transformer' in PyTorch and Zeta. In this implementation we basically take an 2d input tensor representing audio -> then patchify it -> linear proj -> then position embeddings -> then attention and feedforward in a loop for layers. Please Join Agora and tag me if this could be improved in any capacity.

## Install
`pip3 install ast-torch`

## Usage

```python
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

```


# Citation
```bibtex
@misc{gong2021ast,
    title={AST: Audio Spectrogram Transformer}, 
    author={Yuan Gong and Yu-An Chung and James Glass},
    year={2021},
    eprint={2104.01778},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}

```

# License
MIT