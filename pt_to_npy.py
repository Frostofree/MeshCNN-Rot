import torch 
import numpy as np
import os
from tqdm import tqdm

# Load the features from the folder

folder = 'features/M40_heavy_part_train_30/train_features'

for file in tqdm(os.listdir(folder)):
    if file.endswith(".pt"):
        # print(os.path.join("./test_features", file))
        features = torch.load(os.path.join(folder, file))
        features = features.numpy()
        
        # Save the features as numpy arrays
        new_file = os.path.join(folder, file[:-3]+'.npy')
        np.save(new_file, features)
