import os
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform

        # Read image filenames and labels from the txt file
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            self.data = [line.strip().split() for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        img_name = line[0]
        label = int(line[1])  # Convert the second column to an integer

        # Find image that matches with label
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
