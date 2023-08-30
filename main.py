import time, os, torch, math, torch.onnx
from torch import nn

import data, model

device = torch.device('mps' if torch.has_mps else 'cpu')

seed = 81

data_path = "./data/WMT16"

torch.manual_seed(seed)

corpus = data.Corpus(data_path)
