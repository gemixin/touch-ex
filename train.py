# import torchvision
from data.dataset import TouchFLDataset
from torch.utils.data import DataLoader
import data.utils as utils

# --- Setup --- #

RANDOM_STATE = 52  # Set a random seed for reproducibility
SPLIT_SIZE = 0.30  # Proportion of the dataset to use for validation and test splits
STRATIFY_LABEL = 'object'  # The label to use for stratification when splitting the dataset
BATCH_SIZE = 32  # The number of samples in each batch
TRANSFORM_NAME = 'center_crop_224'  # The name of the transform to use
NORM_CACHE_PATH = 'configs/norm_cache.json'  # The path to the normalization cache file
BG_PATH = None  # Set to None to not use background subtraction
NORM = True  # Whether to normalise the images using calculated mean and std
SHUFFLE_MAP = {'train': True, 'val': False, 'test': False}  # Shuffle only the training data

# --- Load and prepare the dataset --- #

# Load the HuggingFace dataset into a DataFrame
df = utils.load_touch_fl_dataset()

# Split the dataset by interaction_id into train, val, and test sets
train_df, val_df, test_df = utils.split_by_interaction(
    df,
    split_size=SPLIT_SIZE,
    stratify_label=STRATIFY_LABEL,
    random_state=RANDOM_STATE,)
split_dfs = {'train': train_df, 'val': val_df, 'test': test_df, }

# Get label info (mappings and materials list) from the training set
label_info = utils.get_label_info(train_df)
idx2label = label_info['idx2label']
materials_list = label_info['materials_list']

# If normalisation is enabled
if NORM:
    # Define dataset configuration parameters for normalisation
    # If these change, norm_stats will be recalculated and the cache updated
    dataset_config = {
        'random_state': RANDOM_STATE,
        'split_size': SPLIT_SIZE,
        'stratify_label': STRATIFY_LABEL,
        'transform_name': TRANSFORM_NAME,
        'bg_path': BG_PATH,
    }
    # Get normalisation stats (mean and std) for the training set, using cache if available
    norm_stats = utils.get_norm_stats(train_df, NORM_CACHE_PATH, dataset_config)
else:
    norm_stats = None

# Create a TouchFLDataset for each split, passing in the appropriate parameters
datasets = {
    split: TouchFLDataset(dataframe=df_split,
                          label_info=label_info,
                          transform_name=TRANSFORM_NAME,
                          norm_stats=norm_stats,
                          bg_path=BG_PATH)
    for split, df_split in split_dfs.items()
}

# Create a DataLoader for each dataset split, using the specified settings
dataloaders = {
    split: DataLoader(datasets[split], batch_size=BATCH_SIZE, shuffle=SHUFFLE_MAP[split])
    for split in split_dfs
}

# train_dataloader = dataloaders['train']
# val_dataloader = dataloaders['val']
# test_dataloader = dataloaders['test']
