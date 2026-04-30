import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data.utils import process_tactile_image


class TouchEXDataset(Dataset):
    """
    A custom PyTorch Dataset class for the Touch-EX dataset, designed to handle both image
    data and multiple types of labels.

    Author: Gemma McLean
    Date: April 2026
    """

    def __init__(self, dataframe, label_info, transform_name,
                 norm_stats=None, bg_path=None):
        """
        Initialise the TouchEXDataset.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the dataset.
            label_info (dict): A dictionary containing label mappings and materials list.
            transform_name (str): The name of the transform to apply to the images.
            norm_stats (dict, optional): A dictionary containing mean and std for
            normalisation. Defaults to None, meaning no normalisation will be applied.
            bg_path (str, optional): Path to the background image for subtraction.
            Defaults to None, meaning no background subtraction will be applied.
        """

        # Initialise with the provided dataset dataframe
        self.df = dataframe.reset_index(drop=True)

        # Extract label info
        self.label2idx = label_info['label2idx']
        self.material_classes = label_info['materials_list']

        # Store the name of the transform to apply to the images
        self.transform_name = transform_name

        # Store the normalisation stats (mean and std)
        self.norm_stats = norm_stats

        # Encode string labels using the provided mappings
        # Store the class mappings in the dataframe
        for label in self.label2idx.keys():
            mapping = self.label2idx[label]
            self.df[f'{label}_class'] = self.df[label].map(mapping)

        # If a background image path is provided, load and preprocess it
        if bg_path is not None:
            bg_img = Image.open(bg_path).convert('RGB')
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
        img_data = row['image']

        # Process the image (including transform and optional background subtraction)
        img_tensor = process_tactile_image(img_data, self.transform_name, self.bg_tensor)

        # Normalise if stats are provided
        if self.norm_stats is not None:
            normalise = transforms.Normalize(mean=self.norm_stats[0],
                                             std=self.norm_stats[1])
            img_tensor = normalise(img_tensor)

        # --- Labels --- #

        # Numeric values - convert to tensors
        force_level_val = torch.tensor(row['force_level'], dtype=torch.long)
        force_n_val = torch.tensor(row['force_n'], dtype=torch.float32)
        fsr_voltage_val = torch.tensor(row['fsr_voltage'], dtype=torch.float32)

        # Text values - keep as strings
        description_value = row['description']
        interaction_num_value = row['interaction_num']
        frame_num_value = row['frame_num']
        interaction_id_value = row['interaction_id']

        # Encode the materials as a multi-hot vector based on the materials list
        materials_vector = torch.zeros(len(self.material_classes), dtype=torch.float32)
        # Get current materials for this sample (may be multiple, separated by ", ")
        current_materials = row['materials'].split(', ')
        # Set the corresponding indices in the vector to 1.0 for each material present
        for mat in current_materials:
            if mat in self.material_classes:
                mat_idx = self.material_classes.index(mat)
                materials_vector[mat_idx] = 1.0

        # Combine features into a single dictionary
        features = {
            'image': img_tensor,
            'force_level': force_level_val,
            'force_n': force_n_val,
            'fsr_voltage': fsr_voltage_val,
            'description': description_value,
            'interaction_num': interaction_num_value,
            'frame_num': frame_num_value,
            'interaction_id': interaction_id_value,
            'materials': materials_vector,
        }

        # Categorical labels (encoded as class indices)
        # Add each categorical label to the features dictionary
        for label in self.label2idx.keys():
            features[label] = torch.tensor(row[f'{label}_class'], dtype=torch.long)

        # Return a dictionary containing everything
        # This makes it easy to swap out task heads later
        return features
