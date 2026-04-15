import torch
import torch.version
import torchvision

# Test torch
print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.rand(1000, 1000).to(device)
y = torch.matmul(x, x)

print(y.device)

# Test torchvision
print(torchvision.__version__)
