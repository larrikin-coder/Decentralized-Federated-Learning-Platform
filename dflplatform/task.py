import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
from collections import OrderedDict
from torchvision import transforms
import pandas as pd

# class MRINet(nn.Module):
#     """Simple CNN for MRI slice classification"""

#     def __init__(self, num_classes=2):
#         super(MRINet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.fc1 = nn.Linear(64 * 64 * 64, 128)  # assuming 128x128 slices
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
class LinearRegressionModel(nn.Module):
    def __init__(self,input_dim,output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.linear(x)
    
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return torch.tensor(self.x[idx],torch.tensor(self.y[idx]))
    

class SalaryDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df["YearsExperience"].values.astype(np.float32).reshape(-1, 1)
        self.y = df["Salary"].values.astype(np.float32).reshape(-1, 1)
        # Normalize features and labels to roughly [0,1] range
        self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
        self.y = (self.y - self.y.min()) / (self.y.max() - self.y.min())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    
    

def load_salary_data(csv_path, batch_size=8, shuffle=True):
    dataset = SalaryDataset(csv_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    testloader = DataLoader(test_ds, batch_size=batch_size)
    return trainloader, testloader

def train_model(model, trainloader, instructions: dict, device):
    model.to(device)
    epochs = instructions.get("epochs", 1)
    lr = instructions.get("lr", 0.01)
    optimizer_type = instructions.get("optimizer_type", "SGD")
    optimizer_params = instructions.get("optimizer_params", {"momentum": 0.9})

    criterion = nn.MSELoss()

    if optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, **optimizer_params)
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, **optimizer_params)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in trainloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return loss.item()

def test_model(model, testloader, device):
    model.to(device)
    criterion = nn.MSELoss()
    loss = 0
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in testloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss += criterion(outputs, y_batch).item()
    return loss / len(testloader)


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)



# class MRIDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nii")]
#         self.transform = transform

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         img_path = self.files[idx]
#         img_obj = nib.load(img_path)
#         img_data = img_obj.get_fdata()

#         # take middle slice
#         slice_ = img_data[:, :, img_data.shape[2] // 2]
#         slice_ = np.array(slice_, dtype=np.float32)

#         # normalize
#         slice_ = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_))
#         slice_ = np.expand_dims(slice_, axis=0)  # add channel dimension (1, H, W)

#         label = 0 if "healthy" in img_path else 1  # simple binary classification

#         if self.transform:
#             slice_ = self.transform(slice_)

#         return torch.tensor(slice_), torch.tensor(label, dtype=torch.long)


# def load_data(partition_dir):
#     transform = transforms.Compose([])
#     dataset = MRIDataset(partition_dir, transform=transform)

#     # Split into train/test
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

#     trainloader = DataLoader(train_ds, batch_size=8, shuffle=True)
#     testloader = DataLoader(test_ds, batch_size=8)

#     return trainloader, testloader


# def train(net, trainloader, epochs, device):
#     net.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#     net.train()
#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = net(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#     return loss.item()


# def test(net, testloader, device):
#     net.to(device)
#     criterion = nn.CrossEntropyLoss()
#     loss, correct, total = 0, 0, 0
#     net.eval()
#     with torch.no_grad():
#         for images, labels in testloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return loss / len(testloader), correct / total


# def get_weights(net):
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]


# def set_weights(net, parameters):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)
