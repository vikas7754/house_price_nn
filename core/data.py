import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class HousePriceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        # Select features and target
        self.features = self.df[[
            'Avg. Area Income',
            'Avg. Area House Age',
            'Avg. Area Number of Rooms',
            'Avg. Area Number of Bedrooms',
            'Area Population'
        ]].values.astype('float32')
        
        self.targets = self.df['Price'].values.astype('float32').reshape(-1, 1)
        
        # Simple normalization (optional but good practice)
        self.mean = self.features.mean(axis=0)
        self.std = self.features.std(axis=0)
        self.features = (self.features - self.mean) / (self.std + 1e-6)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

def get_dataloader(csv_file='housing.csv', batch_size=32):
    dataset = HousePriceDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
