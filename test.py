import torch
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

from CNN import CNNModel
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 100

emnist_dataset_Test = EMNIST(root='data/', split='byclass', train=False, transform=transforms.ToTensor(), download=True)

emnist_dataloader_Test = DataLoader(emnist_dataset_Test, batch_size=batch_size, shuffle=True)


# Model Load
model = torch.load('model.pth')

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

# Model Test with Test Image
data_number = 10
with torch.no_grad() :
    sample_images = random.sample(list(emnist_dataset_Test), data_number)

    for images, labels in sample_images :
        # Print label
        print("Label:", labels)

        # Image visualize
        plt.imshow(images[0], cmap='gray')
        plt.show()

        # Predict Number with Model
        images = images.unsqueeze(0).to(device)
        labels = labels

        outputs = model(images).to(device)
        _, predicted = torch.max(outputs.data, 1)
        print("Predicted:", predicted.item())
        print()