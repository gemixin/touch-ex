from torchvision import transforms


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
