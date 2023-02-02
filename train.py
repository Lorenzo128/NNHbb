import json
import numpy as np
import torch
import torch.optim as optim
from  dataloader import RootDataset
import uproot3
from torch.utils.data import Dataset, DataLoader, random_split
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
aux_branches = config["aux_branches"]
target_branch = config["target_branch"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]

# Load data from ROOT file into PyTorch Dataset
print("======================  RootDataset  =======================")
dataset = RootDataset(root_file, tree_name, input_branches, aux_branches, target_branch)

# Define DataLoader for loading the data in batches
print("======================  DataLoader  =======================")
dataloader = DataLoader(dataset.data, batch_size=batch_size, shuffle=True)

num_samples = len(dataset)
split = [int(0.7 * num_samples), int(0.2 * num_samples), num_samples-int(0.2 * num_samples)-int(0.7 * num_samples)]
#split = [10,10,num_samples-20]
train_dataset, val_dataset, test_dataset = random_split(dataset, split)

# create dataloaders for the training, validation and test datasets
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = HbbClassifier()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    # evaluate the model on the validation data
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()
        val_loss = val_loss / len(val_dataloader)
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Validation Loss: {val_loss}")
