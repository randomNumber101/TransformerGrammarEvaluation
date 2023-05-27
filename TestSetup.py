import random
import torch

x = torch.tensor([[1,3,4], [1,2,3]])
print(x.size())
print(x[0,:])
print(x[1,:])
print(str(x.size(1)) + str(x.size(0)))

for x in range(10):
    print(random.randrange(2))
