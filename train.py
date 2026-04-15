import torchvision
from data.dataset import TouchFLDataset
from torch.utils.data import DataLoader
import data.utils as utils

# Define some constants for the training setup
TARGET_LABEL = 'object'
BATCH_SIZE = 32
BG_PATH = None  # Set to None to not use background subtraction
NORM = False  # Whether to normalise the images using calculated mean and std
SHUFFLE_MAP = {'train': True, 'val': False, 'test': False}  # Shuffle only the training data

# --- Load and prepare the dataset --- #

# Define dataset configuration parameters for splitting and normalisation
dataset_config = {
    'random_state': 52,
    'split_size': 0.30,
    'stratify_label': TARGET_LABEL,
    'bg_path': BG_PATH,
}

# Load the HuggingFace dataset into a DataFrame
df = utils.load_touch_fl_dataset()

# Split the dataset by interaction_id into train, val, and test sets
train_df, val_df, test_df = utils.split_by_interaction(
    df,
    split_size=dataset_config['split_size'],
    stratify_label=dataset_config['stratify_label'],
    random_state=dataset_config['random_state'])
split_dfs = {'train': train_df, 'val': val_df, 'test': test_df, }

# Get label info (mappings and materials list) from the training set
label_info = utils.get_label_info(train_df)
idx2label = label_info['idx2label']
materials_list = label_info['materials_list']

# If normalisation is enabled, get normalisation stats from cache or calculate from
# training data
if NORM:
    norm_stats = utils.get_norm_stats(train_df, dataset_config)
else:
    norm_stats = None

# Create a TouchFLDataset for each split, passing in the appropriate parameters
datasets = {
    split: TouchFLDataset(df_split, label_info, norm_stats,
                          bg_path=dataset_config['bg_path'])
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

# Get a batch of training data and visualise the images with their labels
viz_dataloader = DataLoader(datasets['train'], batch_size=4, shuffle=True)
data = next(iter(viz_dataloader))
inputs, labels = data['image'], data[TARGET_LABEL]
grid = torchvision.utils.make_grid(inputs)
utils.display_image(grid, title=[idx2label[TARGET_LABEL][idx.item()] for idx in labels],
                    norm_stats=norm_stats)
