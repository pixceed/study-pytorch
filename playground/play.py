import torch
import torch.nn as nn

torch.manual_seed(1)

fc = nn.Linear(3, 2)

print(fc.weight)