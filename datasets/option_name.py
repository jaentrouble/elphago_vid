import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import random
import numpy as np
from torchvision import transforms

class OptionNameDataset(Dataset):
    def __init__(self, base_img_path, options_data_path) -> None:
        super().__init__()
        self.base_img = Image.open(base_img_path)
        self.options_df = pd.read_csv(options_data_path)
        self.options_n = len(self.options_df)
        self.font_list = [
            'NanumGothicBold',
            'batang',
            'NanumGothicBold',
            'gulim'
        ]
        self.font_color_list = [
            (115,108,34),
            (173,162,84),
            (198,184,139),
            (125,115,20)
        ]
        self.font_size_list = list(range(11,16))

        self.n = (len(self.options_df)
                  *len(self.font_list)
                  *len(self.font_color_list)
                  *len(self.font_size_list))
        
        self.transform = transforms.Compose([
            transforms.GaussianBlur(3, sigma=(0.01, 2.0)),
            transforms.RandAugment(),
        ])
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        base_img_copy = self.base_img.copy()
        idx = idx % self.n
        font_idx = idx % len(self.font_list)
        idx = idx // len(self.font_list)
        font_color_idx = idx % len(self.font_color_list)
        idx = idx // len(self.font_color_list)
        font_size_idx = idx % len(self.font_size_list)
        idx = idx // len(self.font_size_list)
        option_idx = idx % len(self.options_df)
        idx = idx // len(self.options_df)

        option = self.options_df.loc[option_idx, 'option_name']
        font = ImageFont.truetype(self.font_list[font_idx], self.font_size_list[font_size_idx])
        draw = ImageDraw.Draw(base_img_copy)
        _, _, text_width, text_height = draw.textbbox((0,0), option, font)
        offset_max_w = max(105-text_width, 1)
        offset_max_h = max(20-text_height, 1)
        position = (random.randrange(0, offset_max_w), random.randrange(0, offset_max_h))
        draw.text(position, option, font=font, fill=self.font_color_list[font_color_idx])
        img = torch.from_numpy(np.array(base_img_copy)).permute(2,0,1)
        img = self.transform(img)
        img = img.to(torch.float32)/ 255

        return img, option_idx  