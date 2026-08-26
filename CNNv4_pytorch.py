import os
# 设置环境变量解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 输出尺寸: 32x16x16
        x = self.pool(torch.relu(self.conv2(x)))  # 输出尺寸: 64x8x8
        x = x.view(-1, 64 * 8 * 8)                # 展平
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_data(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "data")
    
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test_epoch(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size=128)
    
    # 初始化模型
    model = CNN().to(device)
    
    # 初始化训练组件
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建TensorBoard日志记录器
    writer_dir = os.path.join(script_dir, "runs/cifar10_pytorch_experiment")
    writer = SummaryWriter(writer_dir)
    
    # 训练参数
    EPOCHS = 10
    best_acc = 0.0
    model_save_path = os.path.join(script_dir, "model/cifar10_cnn_pytorch.pth")
    
    # 训练循环
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
        
        print(f"第 {epoch+1}/{EPOCHS} 轮训练")
        print(f"训练损失: {train_loss:.4f}  准确率: {train_acc:.2f}%")
        print(f"测试损失: {test_loss:.4f}  准确率: {test_acc:.2f}%\n")
        
        # 记录到TensorBoard
        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Training Accuracy', train_acc, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', test_acc, epoch)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型，准确率: {best_acc:.2f}%\n")
    
    writer.close()
    
    # 加载最佳权重并评估
    model.load_state_dict(torch.load(model_save_path))
    final_test_loss, final_test_acc = test_epoch(model, device, test_loader, criterion)
    print(f"最终测试准确率: {final_test_acc:.2f}%")
    print(f"模型保存于: {model_save_path}")

if __name__ == "__main__":
    main()