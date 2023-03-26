import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
import random

class EnchantNDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.imgs = []
        self.labels = []
        for i in range(14):
            for p in self.data_dir.joinpath(f'{i+1}').iterdir():
                self.imgs.append(read_image(str(p))[1:2])
                self.labels.append(i)
        self.transform = transforms.Resize((16,32))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rand_left = random.randint(0, 10)
        rand_right = random.randint(1, 10)
        rand_top = random.randint(0, 10)
        rand_bottom = random.randint(1, 10)
        cut_img = self.imgs[idx][:,rand_top:-rand_bottom,rand_left:-rand_right]
        cut_img = cut_img.to(torch.float32)/ 255
        return self.transform(cut_img), self.labels[idx]
    
