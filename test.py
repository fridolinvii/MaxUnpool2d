import torch
import torch.nn as nn

x = torch.load("x.pt")
indices = torch.load("indices.pt")

unpool = nn.MaxUnpool2d(kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))


###############################################

output_1 = unpool(x,indices)
output_2 = unpool(x,indices)


############################################
print("Sum of Output_1:")
print(sum(sum(sum(sum(output_1)))))
print("---------------------")
print("Sum of Output_2:")
print(sum(sum(sum(sum(output_2)))))


