from torchvision.io import read_image
import numpy as np
import tqdm
from pathlib import Path

for i in tqdm.trange(132, ncols=80):
    img_dir = Path(f'data/advice_gen/{i}')
    images = []
    for img_path in tqdm.tqdm(list(img_dir.iterdir()), ncols=80, leave=False, desc=f'{i+1}/132'):
        images.append(read_image(str(img_path)).numpy())
    images = np.stack(images, axis=0)
    np.save(f'data/advice_gen/{i}.npy', images)
        