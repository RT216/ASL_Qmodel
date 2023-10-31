import torch
import torchvision
import torchvision.models as model
import torch.nn as nn
resnet18 = model.resnet18()
resnet18.fc = nn.Linear(512, 11)
print(resnet18.state_dict().keys())