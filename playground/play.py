import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

fc = nn.Linear(3, 2)

print(fc.weight)
print(fc.bias)

x = torch.tensor([[1, 2, 3]], dtype=torch.float32)

print(type(x))
print()

u = fc(x)

print(u)


# ReLU 関数
print()
print("reru")
z = F.relu(u)
print(z)