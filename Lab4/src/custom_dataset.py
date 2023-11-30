import os
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_directory, text_file, transform=None):
        self.root_directory = root_directory
        self.text_file = text_file
        self.transform = transform

        self.data = []
        try:
            text_file_path = os.path.join(self.root_directory, self.text_file)
            with open(text_file_path, 'r') as file:
                for line in file:
                    self.data.append(line.strip().split())
        except FileNotFoundError:
            print(f"ERROR: File {text_file} not found!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label, _ = self.data[idx]
        label = int(label)
        image_path = os.path.join(self.root_directory, image_name)

        try:
            image = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, IOError):
            print(f"ERROR: Failed to open image: {image_path}!")

        if self.transform:
            image = self.transform(image)

        return image, int(label)
