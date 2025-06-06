import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # patchified stem
        self.stem = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=4, padding=1) # -> 56 x 56

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # 28 x 28
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 14 x 14
        self.bn2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 14 * 14, 128)

        self.fc2 = nn.Linear(128, 8) # x1, y1, x2, y2, x3, y3, x4, y4

        

    def forward(self, x):
        x = self.stem(x)

        x = self.bn1(self.conv1(x))
        x = F.relu(x)

        x = self.bn2(self.conv2(x))
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x