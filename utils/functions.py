import json
import numpy as np
import uproot3 
import torch


def map_targets(target):
        if (target == "data"):
            return -1
        elif "ZZ" in target:
            return 1
        elif "WZ" in target:
            return 1
        elif "WW" in target:
            return 1
        elif "WlvH" in target:
            return 0
        elif "ZvvH" in target:
            return 0
        elif "ZllH" in target:
            return 0
        elif "stop" in target:
            return 2
        elif "ttbar" in target:
            return 2
        elif "Z" in target:
            return 2
        elif "W" in target:
            return 2
        else:
            return -1
