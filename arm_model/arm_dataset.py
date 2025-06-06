import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ArmDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = []
        self.transform = transform

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                image_path = row[0]
                keypoints = list(map(int, row[-8:])) # Only the last 8 points are arm points
                self.data.append((image_path, keypoints))
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        image_path, keypoints = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # Normalize keypoints
        normalized_keypoints = []
        for i in range(len(keypoints)):
            if i % 2 == 0:
                normalized_keypoints.append(keypoints[i] / width)
            else:
                normalized_keypoints.append(keypoints[i] / height)

        normalized_keypoints = torch.tensor(normalized_keypoints, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        return image, normalized_keypoints
