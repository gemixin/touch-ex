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
        self,
        dataframe,
        train_label_mappings,
        test_unseen_label_mappings,
        transform_name,
        norm_stats=None,
        bg_path=None,
        test_unseen=False,
    ):
        """
        Initialise the TouchExDataset object.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the dataset.
            train_label_mappings (dict): A dictionary containing categorical label mappings
                for the training set.
            test_unseen_label_mappings (dict): A dictionary containing categorical label
                mappings for the unseen test set.
            transform_name (str): The name of the transform to apply to the images.
            norm_stats (dict, optional): A dictionary containing mean and std for
                normalisation. Defaults to None, meaning no normalisation will be applied.
            bg_path (str, optional): Path to the background image for subtraction.
                Defaults to None, meaning no background subtraction will be applied.
            test_unseen (bool, optional): Whether the dataset is for the unseen test split.
        """

        # Initialise with the provided dataset dataframe
        self.df = dataframe.reset_index(drop=True)

        # Extract label info
        self.train_label2idx = train_label_mappings["label2idx"]
        self.test_unseen_label2idx = test_unseen_label_mappings["label2idx"]

        # Store the name of the transform to apply to the images
        self.transform_name = transform_name

        # Store the normalisation stats (mean and std)
        self.norm_stats = norm_stats

        # Store whether this dataset is for the unseen test split
        self.test_unseen = test_unseen

        # Define the expected label targets for the unseen test split, mapping them to their
        # corresponding training label spaces
        self.expected_label_targets = {
            "expected_object": "object",
            "expected_object_region": "object_region",
        }

        # Get the appropriate label mappings based on whether this is the unseen test split
        if self.test_unseen:
            self.label2idx = self.test_unseen_label2idx
        else:
            self.label2idx = self.train_label2idx

        # Encode categorical string labels using the provided mappings
        # ["object", "region", "object_region", "force_level", "motion"]
        # Store the class mappings in the dataframe
        for label in self.label2idx.keys():
            mapping = self.label2idx[label]
            # Map the string labels to class indices, filling any unmapped values with -1
            # This allows us to handle any potential unseen labels gracefully
            encoded = self.df[label].map(mapping).fillna(-1).astype("int64")
            self.df[f"{label}_class"] = encoded

        # If this is the unseen test split, also encode the expected labels using
        # the corresponding training-set label spaces.
        if self.test_unseen:
            for expected_label, train_label in self.expected_label_targets.items():
                mapping = self.train_label2idx[train_label]
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

        # Expected labels for the unseen test split are encoded in the training label spaces
        if self.test_unseen:
            for expected_label in self.expected_label_targets:
                features[expected_label] = torch.tensor(
                    row[f"{expected_label}_class"], dtype=torch.long
                )

        # Return a dictionary containing everything
        # This makes it easy to swap out task heads later
        return features
