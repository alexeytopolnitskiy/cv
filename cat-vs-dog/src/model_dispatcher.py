import torch
import torch.nn as nn

class CatDogNN(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.conv1 = nn.Conv2d(self.in_features, 5, 5)
    self.bn1 = nn.BatchNorm2d(5)
    self.conv2 = nn.Conv2d(5, 8, 5)
    self.bn2 = nn.BatchNorm2d(8)
    self.conv3 = nn.Conv2d(8, 16, 5)
    self.bn3 = nn.BatchNorm2d(16)
    self.conv4 = nn.Conv2d(16, 32, 5)
    self.bn4 = nn.BatchNorm2d(32)
    self.relu = nn.LeakyReLU()
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(3200, 500)
    self.bn5 = nn.BatchNorm1d(500)
    self.fc2 = nn.Linear(500, 100)
    self.bn6 = nn.BatchNorm1d(100)
    self.fc3 = nn.Linear(100, 10)
    self.bn7 = nn.BatchNorm1d(10)
    self.fc4 = nn.Linear(10, self.out_features)

  def forward(self, input):
    x = self.pool(self.relu(self.bn1(self.conv1(input))))
    x = self.pool(self.relu(self.bn2(self.conv2(x))))
    x = self.pool(self.relu(self.bn3(self.conv3(x))))
    x = self.pool(self.relu(self.bn4(self.conv4(x))))
    x = x.view(x.size(0), -1)
    x = self.relu(self.bn5(self.fc1(x)))
    x = self.relu(self.bn6(self.fc2(x)))
    x = self.relu(self.bn7(self.fc3(x)))
    out = self.fc4(x)
    return out
