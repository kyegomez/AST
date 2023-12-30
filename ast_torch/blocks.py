from torch import Tensor

# transformer


def patch_split_overlap(x: Tensor, patch_size: int) -> Tensor:
    """Split a 2d tensor into patches with overlap

    Args:
        x (Tensor): 2d tensor of shape (batch, dim)
        patch_size (int): patch size

    Returns:
        Tensor: 3d tensor of shape (batch, num_patches, patch_size)

    Example:
    x = torch.randn(2, 16)
    model = patch_split_overlap(x, 4)
    print(model.shape)

    """
    B, C = x.shape
    num_patches = C // patch_size
    x = x[:, : num_patches * patch_size]
    x = x.reshape(B, num_patches, patch_size)
    return x

