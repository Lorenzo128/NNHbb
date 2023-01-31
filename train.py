import json
import numpy as np
import torch
import torch.optim as optim
from  dataloader import RootDataset
import uproot3
from torch.utils.data import Dataset, DataLoader

from models.model import HbbClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F


# Load config options from json file
with open("configs/config.json", "r") as f:
    config = json.load(f)

root_file = config["root_file"]
tree_name = config["tree_name"]
input_branches = config["input_branches"]
auxs_branches = config["auxs_branches"]
target_branch = config["target_branch"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]

# Load data from ROOT file into PyTorch Dataset
dataset = RootDataset(root_file, tree_name, input_branches, auxs_branches, target_branch)

# Define DataLoader for loading the data in batches
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

print(dataloader[0])
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = HbbClassifier()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = moedel(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

