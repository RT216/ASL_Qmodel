from torchvision import models, transforms, datasets
import torch
from torch import nn
from utils import myCNN
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-2)
# for loading existing model
parser.add_argument('--resume', action='store_true', help='enables resume')
parser.add_argument('--check_point', type=str,
                    default='checkpoints/model_cnn_epoch_20.pth')

if __name__ == '__main__':
    # print the args
    args = parser.parse_args()
    print(args)
    
    try:
        os.makedirs("checkpoints")
    except OSError:
        pass

    train_data_path = '/mnt/d/Dataset/asl-alphabet/asl_alphabet_train/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    torch.manual_seed(1)
    num_train_samples = len(train_dataset)
    # num_train_samples = 20000

    val_split = 0.2
    split = int(num_train_samples * val_split)
    indices = torch.randperm(num_train_samples)


    train_subset = torch.utils.data.Subset(train_dataset, indices[split:])
    val_subset = torch.utils.data.Subset(val_dataset, indices[:split])

    print(len(train_subset), len(val_subset))


    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_subset, 
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False
    )
    classes = train_dataloader.dataset.dataset.classes

    # Load or define model
    resume_epoch = 0
    model = myCNN().to(device)
    print(model)
    
    # define loss func and optimiser with learning rate
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_record = []

    # define tran and test(val) process
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            loss_record.append(str(loss))
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    for t in tqdm(range(resume_epoch, args.epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        start_time = time()
        iter_time = time()
        
        train(train_dataloader, model, loss_fn, optimizer)
        test(val_dataloader, model, loss_fn)
        
        torch.save({'epoch': t+1,
                    'state_dict': model.state_dict()},
                   f"checkpoints/model_cnn_epoch_{t+1}.pth")
        torch.save(model, "checkpoints/model_cnn_epoch.pth")
        print(f"Saved PyTorch Model State to model_cnn_epoch_{t+1}.pth")
        
        print(f'Took {time() - iter_time:.3f} seconds')
        iter_time = time()
        
    print("Done!")
    loss_record_output = '\n'.join(i for i in loss_record)
    f = open('loss_record_output.txt',mode = 'w')
    f.writelines(loss_record_output)
