import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import torchvision
from torchvision import transforms, datasets
from torchvision import models

from train import train
from arm_dataset import ArmDataset


seed = 0

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(0)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


arm_dataset = ArmDataset(csv_path="data.csv", transform=transform)

# Split dataset
dataset_size = len(arm_dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    arm_dataset, [train_size, val_size, test_size],
    generator=torch.Generator()
)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def to_device(obj):
    if torch.cuda.is_available():
        obj = obj.to("cuda")

    return obj


model = models.mobilenet_v2()

model.classifier[1] = nn.Linear(model.last_channel, 8)
model = to_device(model)

adamW_params = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "betas": (0.9, 0.999),
    "eps": 1e-8
}

train(model, 15, train_loader, val_loader, test_loader, loss_func=nn.MSELoss(), optimizer_params=adamW_params)