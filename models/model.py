import torch
import torch.nn as nn
import torch.nn.functional as F

class HbbClassifier(nn.Module):
    def __init__(self):
        super(HbbClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


