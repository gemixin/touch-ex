import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data.utils import process_tactile_image
from data.transforms import get_color_jitter, get_horizontal_flip


class TouchExDataset(Dataset):
    """
    A custom PyTorch Dataset class for the Touch-Ex dataset, designed to handle both image
    data and multiple types of labels.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(
        self,
        dataframe,
        split,
        label_lists,
        transform_name,
        train_augmentations,
        norm_stats=None,
        bg_path=None,
    ):
        """
        Initialises the TouchExDataset with the provided parameters.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the dataset.
            split (str): The split to which the dataset belongs.
            label_lists (dict): Training class lists under 'train', mapping label names
                to ordered lists of class values shared across all splits.
            transform_name (str): The name of the transform to apply to the images.
            train_augmentations (dict): Augmentations applied only when split is
                'train'.
            norm_stats (dict, optional): A dictionary containing mean and std for
                normalisation. Defaults to None, meaning no normalisation will be applied.
            bg_path (str, optional): Path to the background image for subtraction.
                Defaults to None, meaning no background subtraction will be applied.
        """

        # Assign the provided parameters to instance variables
        self.df = dataframe.reset_index(drop=True)
        self.split = split
        self.transform_name = transform_name
        self.norm_stats = norm_stats
        self.color_jitter = (
            get_color_jitter(train_augmentations["color_jitter"])
            if split == "train"
            else None
        )
        self.random_resized_crop = (
            train_augmentations["random_resized_crop"] if split == "train" else False
        )
        self.horizontal_flip = (
            get_horizontal_flip(train_augmentations["horizontal_flip"])
            if split == "train"
            else None
        )

        # Every split uses the training class indices.
        self.label2idx = {
            label: {name: index for index, name in enumerate(classes)}
            for label, classes in label_lists["train"].items()
        }
        for label, mapping in self.label2idx.items():
            if split == "test_unseen_related" and label in {"object", "object_region"}:
                self.df[f"{label}_class"] = -1
            else:
                self.df[f"{label}_class"] = (
                    self.df[label].map(mapping).fillna(-1).astype("int64")
                )

        # If a background image path is provided, load and preprocess it
        if bg_path is not None:
            bg_img = Image.open(bg_path).convert("RGB")
            self.bg_tensor = transforms.ToTensor()(bg_img)
        else:
            self.bg_tensor = None

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """

        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset at the specified index, including
        the processed image tensor and all associated labels.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image tensor and all labels for the sample.
        """

        # --- Process the tactile image --- #

        # Access the image data from the DataFrame row
        row = self.df.iloc[idx]
        img_data = row["image"]

        # Process the image (including transform and optional background subtraction)
        img_tensor = process_tactile_image(
            img_data,
            self.transform_name,
            self.bg_tensor,
            color_jitter=self.color_jitter,
            random_resized_crop=self.random_resized_crop,
        )

        # Optionally flip the processed training image before normalisation
        if self.horizontal_flip is not None:
            img_tensor = self.horizontal_flip(img_tensor)

        # Normalise if stats are provided
        if self.norm_stats is not None:
            normalise = transforms.Normalize(
                mean=self.norm_stats[0], std=self.norm_stats[1]
            )
            img_tensor = normalise(img_tensor)

        # --- Labels --- #

        # Numeric values - convert to tensors
        force_n_val = torch.tensor(row["force_n"], dtype=torch.float32)
        fsr_voltage_val = torch.tensor(row["fsr_voltage"], dtype=torch.float32)

        # Text values - keep as strings
        hardness_value = row["hardness"]
        material_value = row["material"]
        description_value = row["description"]
        interaction_num_value = row["interaction_num"]
        frame_num_value = row["frame_num"]
        interaction_id_value = row["interaction_id"]

        # Combine features into a single dictionary
        features = {
            "image": img_tensor,
            "force_n": force_n_val,
            "fsr_voltage": fsr_voltage_val,
            "hardness": hardness_value,
            "material": material_value,
            "description": description_value,
            "interaction_num": interaction_num_value,
            "frame_num": frame_num_value,
            "interaction_id": interaction_id_value,
        }

        # Categorical labels (encoded as class indices)
        # Add each categorical label to the features dictionary
        for label in self.label2idx.keys():
            features[label] = torch.tensor(row[f"{label}_class"], dtype=torch.long)

        # Retain source identities for related-object diagnostic plots.
        if self.split == "test_unseen_related":
            features["source_object"] = row["object"]
            features["source_object_region"] = row["object_region"]

        # Return a dictionary containing everything
        return features
