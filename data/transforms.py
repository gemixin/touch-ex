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


def get_train_augmentation(train_augmentations):
    """
    Build spatial augmentation applied only to preprocessed training images.

    Args:
        train_augmentations (dict): Validated augmentation settings from the data
            configuration.

    Returns:
        torchvision.transforms.Compose: The composed training augmentation transform.
    """

    augmentation_steps = []
    random_crop_padding = train_augmentations["random_crop_padding"]

    if random_crop_padding is not None:
        augmentation_steps.append(
            transforms.RandomCrop(
                (224, 224),
                padding=random_crop_padding,
                padding_mode="reflect",
            )
        )

    return transforms.Compose(augmentation_steps)
