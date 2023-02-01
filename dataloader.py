import json
import numpy as np
import uproot3 
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import utils.functions as uf

class RootDataset(Dataset):
    def __init__(self, root_file, tree_name, input_branches, auxs_branches, target_branch):
        tree = uproot3.open(root_file)[tree_name]

        targets_string = tree.array(target_branch.encode())
        targets_string = np.array([x.decode() for x in targets_string])
        target_mapper = np.vectorize(uf.map_targets)
        self.targets = torch.from_numpy(target_mapper(targets_string)).long()

        inputs_dict = tree.arrays([branch.encode() for branch in input_branches])
        self.inputs = torch.from_numpy(np.stack([inputs_dict[branch.encode()] for branch in input_branches], axis=1)).float()
        
        int_dict = tree.arrays([branch.encode() for branch in auxs_branches["int_branches"]])
        self.auxs_int = torch.from_numpy(np.stack([int_dict[branch.encode()] for branch in auxs_branches["int_branches"]], axis=1)).long()
        float_dict = tree.arrays([branch.encode() for branch in auxs_branches["float_branches"]])
        self.auxs_float = torch.from_numpy(np.stack([float_dict[branch.encode()] for branch in auxs_branches["float_branches"]], axis=1)).float()
        
        self.data = torch.utils.data.TensorDataset(self.inputs, self.targets)
        self.data[torch.where(self.targets!=-1)]


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):

        inputs = self.inputs[idx]
        auxs_int = self.auxs_int[idx]
        auxs_float = self.auxs_float[idx]
        target = self.targets[idx]

        return (torch.from_numpy(inputs).float(), torch.tensor(target).long())
    
