from torchvision import transforms


def get_transform(transform_name):
    """
    Get the specified transform by name.

    Args:
        transform_name (str): The name of the transform to retrieve. Options are
        'pad_224', 'center_crop_224', 'random_crop_224'.

    Returns:
        torchvision.transforms.Compose: The composed transform corresponding to the
        specified name.
    """

    if transform_name == 'pad_224':
        return pad_224()
    elif transform_name == 'center_crop_224':
        return center_crop_224()
    elif transform_name == 'random_crop_224':
        return random_crop_224()
    else:
        raise ValueError(f'Invalid transform name: {transform_name}.')


def pad_224():
    """
    Pad the image to 320x320 maintaining aspect ratio, and then resize to 224x224

    Returns:
        torchvision.transforms.Compose: The composed transform for resizing images.
    """

    return transforms.Compose([
        transforms.Pad([40, 0, 40, 0]),
        transforms.Resize((224, 224))
    ])


def center_crop_224():
    """.
    Resize the shorter side to 256 maintaining aspect ratio, and then take a center crop
    of 224x224.

    Returns:
        torchvision.transforms.Compose: The composed transform for resizing images.
    """

    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224))
    ])


def random_crop_224():
    """
    Resize the shorter side to 256 maintaining aspect ratio, and then take a random crop
    of 224x224.

    Returns:
        torchvision.transforms.Compose: The composed transform for resizing images.
    """

    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop((224, 224))
    ])
