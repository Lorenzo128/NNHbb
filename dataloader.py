import json
import numpy as np
import uproot3 
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RootDataset(Dataset):
    def __init__(self, root_file, tree_name, input_branches, auxs_branches, target_branch):
        tree = uproot3.open(root_file)[tree_name]
        inputs_dict = tree.arrays([branch.encode() for branch in input_branches])
        self.inputs = np.concatenate([inputs_dict[branch.encode()] for branch in input_branches])
        auxs_dict = tree.arrays([branch.encode() for branch in auxs_branches])
        self.auxs = np.concatenate([auxs_dict[branch.encode()] for branch in auxs_branches])
        self.targets = tree.array(target_branch.encode())

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        print(auxs[0], auxs[1], targets[0])
        if idx %100 == 0:
            inputs = self.inputs[idx]
        auxs = self.auxs[idx]
        targets = self.targets[idx]
        if ((auxs[0] < 75) | (auxs[0]>150)):
            return None
        if auxs[2] != 2:
            return None
        if ((auxs[1] < 1.5) | (auxs[1] > 3.5)):
            return None
        if (targets == "data"):
            return None
        elif "ZZ" in targets:
            target = 1
        elif "WZ" in targets:
            target = 1
        elif "WW" in targets:
            target = 1
        elif "WlvH" in targets:
            target = 0
        elif "ZvvH" in targets:
            target = 0
        elif "ZllH" in targets:
            target = 0
        elif "stop" in targets:
            target = 2
        elif "ttbar" in targets:
            target = 2
        elif "Z" in targets:
            target = 2
        elif "W" in targets:
            target = 2
        else:
            return None

        return (torch.from_numpy(inputs).float(), torch.tensor(target).long())

