import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import nibabel as nib   # for NIfTI MRI files (.nii)
import numpy as np
from collections import OrderedDict
from torchvision import transforms


# -------------------------
# 1. Define CNN for MRI slices
# -------------------------
class MRINet(nn.Module):
    """Simple CNN for MRI slice classification"""

    def __init__(self, num_classes=2):
        super(MRINet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # assuming 128x128 slices
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# 2. MRI Dataset Loader
# -------------------------
class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".nii")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img_obj = nib.load(img_path)
        img_data = img_obj.get_fdata()

        # take middle slice
        slice_ = img_data[:, :, img_data.shape[2] // 2]
        slice_ = np.array(slice_, dtype=np.float32)

        # normalize
        slice_ = (slice_ - np.min(slice_)) / (np.max(slice_) - np.min(slice_))
        slice_ = np.expand_dims(slice_, axis=0)  # add channel dimension (1, H, W)

        label = 0 if "healthy" in img_path else 1  # simple binary classification

        if self.transform:
            slice_ = self.transform(slice_)

        return torch.tensor(slice_), torch.tensor(label, dtype=torch.long)


# -------------------------
# 3. Data Loaders
# -------------------------
def load_data(partition_dir):
    transform = transforms.Compose([])
    dataset = MRIDataset(partition_dir, transform=transform)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_ds, batch_size=8, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=8)

    return trainloader, testloader


# -------------------------
# 4. Train/Test helpers
# -------------------------
def train(net, trainloader, epochs, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return loss.item()


def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0, 0, 0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss / len(testloader), correct / total


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
