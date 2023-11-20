import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

from CNN import CNNModel

# Use cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter
learning_rate = 0.001
train_epochs = 15
batch_size = 100


# EMNIST dataset Lost (ByClass)
emnist_dataset = EMNIST(root='data/', split='byclass', train=True, transform=transforms.ToTensor(), download=True)
emnist_dataloader = DataLoader(emnist_dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화 및 손실 함수, 최적화 기준 정의
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(emnist_dataloader)


# 학습
for epoch in range(train_epochs):
    avg_loss = 0
    for images, labels in emnist_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).to(device)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    print(f'Epoch [{epoch + 1}], Loss: {avg_loss}')

# Model Save
torch.save(model, 'model.pth')

# Test Data Loader
emnist_dataset_Test = EMNIST(root='data/', split='byclass', train=False, transform=transforms.ToTensor(), download=True)

emnist_dataloader_Test = DataLoader(emnist_dataset_Test, batch_size=batch_size, shuffle=True)

# Evaluate Model
model.eval()
correct = 0
total = 0

with torch.no_grad() :
    for images, labels in emnist_dataloader_Test :
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('accuracy:', 100 * correct / total)

