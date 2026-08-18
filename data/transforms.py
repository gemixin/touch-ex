"""
A module containing functions for defining and applying image transforms to the tactile
images in the Touch-Ex dataset.

Author: Gemma McLean
Date: April 2026
"""

from torchvision import transforms


def get_transform(transform_name):
    """
    Get the specified transform by name.

    Args:
        transform_name (str): The name of the transform to retrieve. Options are
        'pad_224' or 'center_crop_224'.

    Returns:
        torchvision.transforms.Compose: The composed transform corresponding to the
            specified name.
    """

    if transform_name == "pad_224":
        return _pad_224()
    elif transform_name == "center_crop_224":
        return _center_crop_224()
    else:
        raise ValueError(f"Invalid transform name: {transform_name}.")


def _pad_224():
    """
    Pad the image to 320x320 maintaining aspect ratio, and then resize to 224x224

    Returns:
        torchvision.transforms.Compose: The composed transform for resizing images.
    """

    return transforms.Compose(
        [transforms.Pad([40, 0, 40, 0]), transforms.Resize((224, 224))]
    )


def _center_crop_224():
    """
    Resize the shorter side to 256 maintaining aspect ratio, and then take a center crop
    of 224x224.

    Returns:
        torchvision.transforms.Compose: The composed transform for resizing images.
    """

    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop((224, 224))])


def get_random_resized_crop_224():
    """
    Get the training-only transform that randomly crops and resizes an image to 224x224
    using torchvision's default scale and aspect-ratio ranges.

    Returns:
        torchvision.transforms.Compose: The composed random crop transform.
    """

    return transforms.RandomResizedCrop((224, 224))


def get_color_jitter(color_jitter):
    """
    Build the optional ColorJitter transform applied to source training images.

    Args:
        color_jitter (dict or None): Validated ColorJitter settings.

    Returns:
        torchvision.transforms.ColorJitter or None: The configured transform.
    """

    if color_jitter is None:
        return None

    return transforms.ColorJitter(**color_jitter)


def get_horizontal_flip(horizontal_flip):
    """
    Build the optional horizontal-flip transform applied to training images.

    Args:
        horizontal_flip (float or None): Probability of flipping an image.

    Returns:
        torchvision.transforms.RandomHorizontalFlip or None: The configured transform.
    """

    if horizontal_flip is None:
        return None

    return transforms.RandomHorizontalFlip(p=horizontal_flip)
