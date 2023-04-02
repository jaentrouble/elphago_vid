import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from pathlib import Path
import random
import numpy as np
import tqdm

ONE_OPTION_ADV = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 26, 28, 34, 36, 38, 40, 42, 45, 46, 47, 48, 77, 82]
TWO_OPTION_ADV = [49, 50, 53, 54]

class AdviceDataset(Dataset):
    def __init__(self, data_dir, multiplier) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.data_dir = Path(data_dir)
        self.img_path_list = []
        self.img_lens = []
        for i in range(132):
            self.img_path_list.append(list(self.data_dir.joinpath(f'{i}').iterdir()))
            self.img_lens.append(len(self.img_path_list[-1]))
        self.transform = transforms.Resize((64,288))
        assert len(self.img_path_list) == 132
        
    def __len__(self):
        return 132 * self.multiplier

    def __getitem__(self, idx):
        advice_idx = idx % 132
        img_idx = random.randrange(0,self.img_lens[advice_idx])
        loaded_img = read_image(str(self.img_path_list[advice_idx][img_idx]))

        rand_left = random.randint(0, 10)
        rand_right = random.randint(1, 10)
        rand_top = random.randint(0, 10)
        rand_bottom = random.randint(1, 10)
        cut_img = loaded_img[:,rand_top:-rand_bottom,rand_left:-rand_right]
        cut_img = cut_img.to(torch.float32)/ 255
        return self.transform(cut_img), advice_idx
    
class AdviceOneOptionDataset(Dataset):
    def __init__(self, data_dir) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.img_path_list = []
        self.option_idx = []
        for i, ooa in enumerate(ONE_OPTION_ADV):
            for p in tqdm.tqdm(list(self.data_dir.joinpath(f'{ooa}').iterdir()), 
                               ncols=80, leave=False, 
                               desc=f'{i+1}/{len(ONE_OPTION_ADV)}'):
                self.img_path_list.append(p)
                self.option_idx.append(int(p.stem.split('_')[-2]))

        self.transform = transforms.Resize((64,288))
        assert len(self.img_path_list) == 24*15840
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        loaded_img = read_image(str(self.img_path_list[idx]))
        rand_left = random.randint(0, 10)
        rand_right = random.randint(1, 10)
        rand_top = random.randint(0, 10)
        rand_bottom = random.randint(1, 10)
        cut_img = loaded_img[:,rand_top:-rand_bottom,rand_left:-rand_right]
        cut_img = cut_img.to(torch.float32)/ 255
        return self.transform(cut_img), self.option_idx[idx]