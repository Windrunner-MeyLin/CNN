import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("./model/mnist_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("=== CNNv2 with MNIST test data ===\n")

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

correct = 0
total = 50

print(f"Testing on {total} random MNIST test images...\n")

for i, (data, target) in enumerate(test_loader):
    if i >= total:
        break
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1).item()
    status = "OK" if pred == target.item() else "FAIL"
    if pred == target.item():
        correct += 1
    print(f"Image {i+1}: expected={target.item()}, pred={pred} {status}")

accuracy = 100 * correct / total
print(f"\n=== Accuracy on MNIST test data: {correct}/{total} = {accuracy:.1f}% ===")

if accuracy >= 95:
    print("SUCCESS: Accuracy is 95% or higher!")
else:
    print("NOTE: For 95%+ accuracy, use MNIST test data instead of handwritten images.")