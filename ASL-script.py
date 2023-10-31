# %%
from torchvision import models, transforms, datasets
import torch
from torch import nn
from utils import myCNN
import matplotlib.pyplot as plt

# %%
train_data_path = '/mnt/d/Dataset/asl-alphabet/asl_alphabet_train/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
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

# %%
%%time
train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)

# %%
%%time
val_dataset = datasets.ImageFolder(train_data_path, transform=test_transforms)

# %%
torch.manual_seed(0)
num_train_samples = len(train_dataset)
# num_train_samples = 20000

val_split = 0.2
split = int(num_train_samples * val_split)
indices = torch.randperm(num_train_samples)


train_subset = torch.utils.data.Subset(train_dataset, indices[split:])
val_subset = torch.utils.data.Subset(val_dataset, indices[:split])

len(train_subset), len(val_subset)

# %%
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

# %%
classes = train_dataloader.dataset.dataset.classes

# %%
for img, label in train_dataloader:
    print(img.shape, label.shape)
    print(f'Ground Truth {classes[label[0]]}')
    print(img[0].size())
    print(img[0].permute(1, 2, 0).size())
    plt.imshow(img[0].permute(1, 2, 0),cmap='gray')
    break

# %%
model = myCNN().to(device)
print(model)

# %%
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# %%
from time import time
from tqdm import tqdm

def train(model,
          criterion,
          optimizer,
          train_dataloader,
          test_dataloader,
          print_every,
          num_epoch):
    steps = 0
    train_losses, val_losses = [], []

    model.to(device)
    for epoch in tqdm(range(num_epoch)):
        running_loss = 0
        correct_train = 0
        total_train = 0
        start_time = time()
        iter_time = time()
        
        #model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            steps += 1
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            
            output = model(images)
            loss = criterion(output, labels)
            
            correct_train += (torch.max(output, dim=1)[1] == labels).type(torch.float).sum().item()
            total_train += labels.size(0)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Logging
            if steps % print_every == 0:
                print(f'Epoch [{epoch + 1}]/[{num_epoch}]. Batch [{i + 1}]/[{len(train_dataloader)}].', end=' ')
                print(f'Train loss {running_loss / steps:.5f}.', end=' ')
                print(f'Train acc {correct_train / total_train * 100:.5f}.', end=' ')
                with torch.no_grad():
                    # model.eval()
                    correct_val, total_val = 0, 0
                    val_loss = 0
                    for images, labels in test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        val_loss += loss.item()

                        correct_val += (torch.max(output, dim=1)[1] == labels).type(torch.float).sum().item()
                        total_val += labels.size(0)

                print(f'Val loss {val_loss / len(test_dataloader):.5f}. Val acc {correct_val / total_val * 100:.5f}.', end=' ')
                print(f'Took {time() - iter_time:.5f} seconds')
                iter_time = time()

                train_losses.append(running_loss / total_train)
                val_losses.append(val_loss / total_val)
        scheduler.step(val_loss / len(test_dataloader))


        print(f'Epoch took {time() - start_time}') 
        torch.save(model, f'checkpoints/checkpoint_{correct_val / total_val * 100:.2f}.pth')
        
    return model, train_losses, val_losses

# %%
print_every = 50
num_epoch = 100

model, train_losses, val_losses = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    test_dataloader=val_dataloader,
    print_every=print_every,
    num_epoch=num_epoch
)

# %%
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()

# %%
from pathlib import Path
from PIL import Image


test_data_path = Path('/mnt/d/Dataset/asl-alphabet/asl_alphabet_test')


class ASLTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, transforms=None):
        super().__init__()
        
        self.transforms = transforms
        self.imgs = sorted(list(Path(root_path).glob('*.jpg')))
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('L')
        
        label = img_path.parts[-1].split('_')[0]
        if self.transforms:
            img = self.transforms(img)
        
        return img, label

# %%
test_dataset = ASLTestDataset(test_data_path, transforms=test_transforms)

columns = 7
row = round(len(test_dataset) / columns)

fig, ax = plt.subplots(row, columns, figsize=(columns * row, row * columns))
plt.subplots_adjust(wspace=0.1, hspace=0.2)

#test_model = torch.load("checkpoints/checkpoint_87.08.pth", map_location='cpu')
test_model = model
test_model.to(device)

i, j = 0, 0
for img, label in test_dataset:
    img = torch.Tensor(img)
    img = img.to(device)
    test_model.eval()
    prediction = test_model(img[None])

    ax[i][j].imshow(img.cpu().permute(1, 2, 0),cmap='gray')
    ax[i][j].set_title(f'GT {label}. Pred {classes[torch.max(prediction, dim=1)[1]]}') #torch.max(prediction, dim=1)[1]
    ax[i][j].axis('off')
    j += 1
    if j == columns:
        j = 0
        i += 1
        
plt.show()


