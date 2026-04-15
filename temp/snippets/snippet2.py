import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class TouchFLDataset(Dataset):
    def __init__(self, dataframe, bg_image_path, material_classes):
        """
        Args:
            dataframe: Your Pandas DataFrame (e.g., train_df, val_df, or test_df)
            bg_image_path: Path to 'baseline_empty.jpg'
            material_classes: List of all unique material strings (to create multi-hot vectors)
        """
        self.df = dataframe.reset_index(drop=True)
        self.material_classes = material_classes
        
        # 1. Load the background image ONCE into memory as a float tensor (0.0 to 1.0)
        bg_img = Image.open(bg_image_path).convert("RGB")
        self.bg_tensor = transforms.ToTensor()(bg_img)
        
        # 2. Define the transforms applied AFTER background subtraction
        # Assuming images are 240 (width) x 320 (height). 
        # Pad left/right by 40 to make it 320x320, then resize to 224x224.
        self.post_process = transforms.Compose([
            transforms.Pad([40, 0, 40, 0]), 
            transforms.Resize((224, 224), antialias=True)
        ])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # --- 1. IMAGE PROCESSING ---
        row = self.df.iloc[idx]
        
        # Load the tactile image and convert to float tensor
        # (Assuming you downloaded the images locally or have a path column)
        tactile_path = row["image_path"] 
        tactile_img = Image.open(tactile_path).convert("RGB")
        tactile_tensor = transforms.ToTensor()(tactile_img)
        
        # Subtract the background (Values will safely range from -1.0 to 1.0)
        diff_tensor = tactile_tensor - self.bg_tensor
        
        # Pad and Resize the isolated deformation
        final_image = self.post_process(diff_tensor)
        
        
        # --- 2. LABEL PROCESSING ---
        
        # A. Object Classification (Single Class integer)
        # Assuming you mapped string labels to ints like: {"beans": 0, "metal_nut": 1...}
        obj_label = torch.tensor(row["object_encoded"], dtype=torch.long)
        
        # B. Force Regression (Continuous float)
        force_val = torch.tensor(row["force_n"], dtype=torch.float32)
        
        # C. Materials Multi-Label (Multi-hot vector)
        # e.g., Output: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        material_vector = torch.zeros(len(self.material_classes), dtype=torch.float32)
        current_materials = row["materials"].split(", ") # Adjust based on how your data separates multiple materials
        for mat in current_materials:
            if mat in self.material_classes:
                mat_idx = self.material_classes.index(mat)
                material_vector[mat_idx] = 1.0
                
        # Return a dictionary containing everything!
        # This makes it super easy to swap out task heads later.
        return {
            "image": final_image,
            "object": obj_label,
            "force": force_val,
            "materials": material_vector
        }

# How to use this in your training loop:
# Because the __getitem__ returns a dictionary, testing different tasks is incredibly easy. When you write your training loop, you just grab the specific label you want to train on for that experiment:

# for batch in dataloader:
#     images = batch["image"].to(device)
    
#     # If training the Object Classifier:
#     labels = batch["object"].to(device)
#     loss = cross_entropy_loss(predictions, labels)
    
#     # OR, if you decide to swap to training the Force Regressor:
#     # labels = batch["force"].to(device)
#     # loss = mse_loss(predictions, labels)