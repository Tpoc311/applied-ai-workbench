import torch
import torchvision


def decode_image(path: str) -> torch.Tensor:
    """
    PyTorch's decode_image wrapper, which read images even if they are in grayscale or RGBA format.

    :param path: Path to image.
    :return: Image in torch.Tensor form.
    """
    image = torchvision.io.decode_image(path)
    if image.shape[0] == 4:
        image = image[:3, :, :]
    elif image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image
