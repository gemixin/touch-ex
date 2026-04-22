import json
from data.builder import get_dataloaders

# --- Setup --- #

# Get default data config from json file
with open('configs/default_data_config.json', 'r', encoding='utf-8') as f:
    data_config = json.load(f)

# Override any config parameters here if needed
# data_config['split_size'] = 0.20
# data_config['norm_type'] = None

# --- Load and prepare the dataset --- #
dataloaders, label_info = get_dataloaders(data_config)
print(dataloaders['train'].dataset.df.shape)
print(label_info['materials_list'])
