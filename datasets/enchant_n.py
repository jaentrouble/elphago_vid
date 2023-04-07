import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
import random

class EnchantNDataset(Dataset):
    def __init__(self, data_dir, multiplier=100) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.data_dir = Path(data_dir)
        self.imgs = []
        self.labels = []
        for i in range(14):
            for p in self.data_dir.joinpath(f'{i+1}').iterdir():
                self.imgs.append(read_image(str(p)))
                self.labels.append(i)
        self.transform = transforms.Compose([
            transforms.RandAugment(),
            transforms.Resize((16,32)),
        ])
        
    def __len__(self):
        return len(self.imgs)*self.multiplier

    def __getitem__(self, idx):
        transformed_img = self.transform(self.imgs[idx])
        transformed_img = transformed_img.to(torch.float32)/ 255

        return transformed_img, self.labels[idx]
    
