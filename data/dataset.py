import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data.utils import process_tactile_image


class TouchExDataset(Dataset):
    """
    A custom PyTorch Dataset class for the Touch-Ex dataset, designed to handle both image
    data and multiple types of labels.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(
        self, dataframe, split, mappings, transform_name, norm_stats=None, bg_path=None
    ):
        """
        Initialise the TouchExDataset object.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the dataset.
            split (str): The split to which the dataset belongs.
            mappings (dict): A dictionary containing label mappings for different splits.
            transform_name (str): The name of the transform to apply to the images.
            norm_stats (dict, optional): A dictionary containing mean and std for
                normalisation. Defaults to None, meaning no normalisation will be applied.
            bg_path (str, optional): Path to the background image for subtraction.
                Defaults to None, meaning no background subtraction will be applied.
        """

        # Initialise with the provided dataset dataframe
        self.df = dataframe.reset_index(drop=True)

        # Store the split name
        self.split = split

        # Store the name of the transform to apply to the images
        self.transform_name = transform_name

        # Store the normalisation stats (mean and std)
        self.norm_stats = norm_stats

        # Determine if this is an unseen test split based on the split name
        if self.split in ["test_unseen_matched", "test_unseen_related"]:
            self.test_unseen = True
        else:
            self.test_unseen = False

        # Get the appropriate label mappings based on the split
        if self.split == "test_unseen_matched":
            self.label2idx = mappings["test_unseen_matched"]["label2idx"]
        elif self.split == "test_unseen_related":
            self.label2idx = mappings["test_unseen_related"]["label2idx"]
        # Standard train, val and test splits use the training label mappings
        else:
            self.label2idx = mappings["train"]["label2idx"]

        # Encode categorical string labels using the provided mappings
        # ["object", "region", "object_region", "force_level", "motion"]
        # Store the class mappings in the dataframe
        for label in self.label2idx.keys():
            mapping = self.label2idx[label]
            # Map the string labels to class indices, filling any unmapped values with -1
            # This allows us to handle any potential unseen labels gracefully
            encoded = self.df[label].map(mapping).fillna(-1).astype("int64")
            self.df[f"{label}_class"] = encoded

        # Define the expected label targets for the unseen test splits
        self.expected_label_targets = {
            "expected_object": "object",
            "expected_object_region": "object_region",
        }

        # If this is an unseen test split, also encode the expected labels using
        # the corresponding training-set label spaces.
        if self.test_unseen:
            for expected_label, train_label in self.expected_label_targets.items():
                mapping = mappings["train"]["label2idx"][train_label]
                self.df[f"{expected_label}_class"] = (
                    self.df[expected_label].map(mapping).fillna(-1).astype("int64")
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
        img_tensor = process_tactile_image(img_data, self.transform_name, self.bg_tensor)

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

        # Expected labels for the unseen test data are encoded in the training label spaces
        if self.test_unseen:
            for expected_label in self.expected_label_targets:
                features[expected_label] = torch.tensor(
                    row[f"{expected_label}_class"], dtype=torch.long
                )

        # Return a dictionary containing everything
        # This makes it easy to swap out task heads later
        return features
