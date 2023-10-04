import numpy as np
import torch
x = torch.tensor(3., requires_grad=True)
y = torch.tensor(2., requires_grad=False)
z = torch.norm(x ** y, 2)
z.backward()
print(1)
