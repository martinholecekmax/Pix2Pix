from PIL import Image
import numpy as np
import os
import config
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))

        # Channels x Width x Height (Original image is 1200 x 600 - 2 images of 600 x 600)
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
