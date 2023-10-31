from torchvision import models, transforms, datasets
import torch
from torch import nn
from utils import myCNN, train
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import numpy as np

train_data_path = '/mnt/d/Dataset/asl-alphabet/asl_alphabet_train/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(train_data_path, transform=test_transforms)

torch.manual_seed(time())
num_train_samples = len(train_dataset)
# num_train_samples = 20000

val_split = 0.2
split = int(num_train_samples * val_split)
indices = torch.randperm(num_train_samples)

train_subset = torch.utils.data.Subset(train_dataset, indices[split:])
val_subset = torch.utils.data.Subset(val_dataset, indices[:split])

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_subset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=12,
    pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_subset,
    batch_size=4,
    shuffle=False,
    num_workers=12,
    pin_memory=True
)

classes = train_dataloader.dataset.dataset.classes

model= myCNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

print_every = 50
num_epoch = 50
model = torch.load("checkpoints/checkpoint_86.53.pth", map_location='cpu')
model = model.to(device)

train_losses_array = np.load('train_losses_array.npy')
train_losses = train_losses_array.tolist()
val_losses_array = np.load('val_losses_array.npy')
val_losses = val_losses_array.tolist()

# print(train_losses)

model, train_losses, val_losses = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
    print_every=print_every,
    num_epoch=num_epoch
)


train_losses_array=np.array(train_losses)
val_losses_array=np.array(val_losses)
np.save('train_losses_array.npy',train_losses_array) 
np.save('val_losses_array.npy',val_losses_array) 