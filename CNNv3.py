import os
import numpy as np
from PIL import Image

# 设置环境变量解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 工具

# 设置随机种子以保证结果可复现
def set_seed(seed=42):
    """
    设置随机种子，确保每次运行代码时生成的随机数相同。
    这对于实验的可重复性非常重要。
    """
    torch.manual_seed(seed)

# 数据预处理与加载
def load_data(batch_size=64, num_workers=2):
    """
    加载 CIFAR-10 数据集，并进行预处理。
    - ToTensor(): 将图像数据转换为 PyTorch 张量。
    - Normalize(): 对图像数据进行标准化（减去均值，除以标准差）。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像或 NumPy 数组转换为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
    ])

    # 下载训练集trainset和测试集testset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(script_dir, "data")
    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    

    # 下载训练集trainset和测试集testset
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 第一组卷积：输入3通道（RGB图像），输出32通道，3x3卷积核，padding=1保持尺寸
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # 第二组卷积：输入32通道，输出64通道，3x3卷积核，padding=1保持尺寸
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 第三组卷积：输入64通道，输出64通道，3x3卷积核，padding=1保持尺寸
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # 第四组卷积：输入64通道，输出128通道，3x3卷积核，padding=1保持尺寸
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 第五组卷积：输入128通道，输出128通道，3x3卷积核，padding=1保持尺寸
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # 第六组卷积：输入128通道，输出256通道，3x3卷积核，padding=1保持尺寸
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # 最大池化层：2x2窗口，步幅2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一层全连接：输入特征数256*4*4（CIFAR-10图像32x32经过三次池化后变为4x4），输出512
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        # 第二层全连接：输入512，输出num_classes（CIFAR-10为10个类别）
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        
        # Dropout层：随机丢弃50%神经元防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 第一组：卷积1 -> ReLU -> 卷积2 -> ReLU -> 最大池化（32x32x3 -> 32x32x32 -> 32x32x64 -> 16x16x64）
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 第二组：卷积3 -> ReLU -> 卷积4 -> ReLU -> 最大池化（16x16x64 -> 16x16x64 -> 16x16x128 -> 8x8x128）
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # 第三组：卷积5 -> ReLU -> 卷积6 -> ReLU -> 最大池化（8x8x128 -> 8x8x128 -> 8x8x256 -> 4x4x256）
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        
        # 展平成一维向量（256*4*4 = 4096）
        x = x.view(-1, 256 * 4 * 4)
        
        # 第一层全连接 -> ReLU
        x = F.relu(self.fc1(x))
        
        # Dropout层（仅训练时生效）
        x = self.dropout(x)
        
        # 第二层全连接（输出类别概率）
        x = self.fc2(x)
        
        return x


# 创建测试图像
def create_test_images(num_images=10):
    """从CIFAR-10数据集提取测试图像"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建测试图像目录
    test_image_dir = os.path.join(script_dir, "test_images")
    os.makedirs(test_image_dir, exist_ok=True)
    
    # 定义CIFAR-10类别标签
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 加载CIFAR-10测试集
    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(
        root=os.path.join(script_dir, "data"), 
        train=False, 
        download=True, 
        transform=transform
    )
    
    print(f"\n正在创建 {num_images} 张测试图像...")
    
    for i in range(num_images):
        img_tensor, label = testset[i]
        
        # 将Tensor转换为PIL图像
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        # 保存图像，文件名格式: [标签][唯一标识].png
        filename = f"[{classes[label]}][{i}].png"
        filepath = os.path.join(test_image_dir, filename)
        img.save(filepath)
    
    # 同时保存一张为 test_image.png
    img_tensor, label = testset[0]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    single_image_path = os.path.join(script_dir, "test_image.png")
    img.save(single_image_path)
    
    print(f"测试图像已创建完成！")
    print(f"  - 批量测试图像: {test_image_dir}/")
    print(f"  - 单张测试图像: test_image.png")


# 训练模型
def train_model(net, trainloader, criterion, optimizer, device, writer, epochs=10):
    """
    训练模型：
    - 使用交叉熵损失函数和SGD优化器。
    - 每个epoch结束后记录平均损失到 TensorBoard。
    """
    print("Starting Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        net.train()  # 切换到训练模式
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到GPU/CPU

            optimizer.zero_grad()  # 清空梯度
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.3f}")
        writer.add_scalar('Training Loss', avg_loss, epoch)  # 将损失写入 TensorBoard

    print('Finished Training')

# 测试模型
def test_model(net, testloader, device, writer, epoch):
    """
    测试模型：
    - 在测试集上评估模型性能。
    - 将测试准确率写入 TensorBoard。
    """
    correct = 0
    total = 0
    net.eval()  # 切换到评估模式
    with torch.no_grad():  # 不计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    writer.add_scalar('Test Accuracy', accuracy, epoch)  # 将准确率写入 TensorBoard

# 主函数
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    set_seed()  # 设置随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU可用

    # 加载数据
    trainloader, testloader = load_data()

    # 初始化模型、损失函数和优化器
    net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 使用SGD优化器

    # 创建 TensorBoard 的日志记录器
    writer_dir = os.path.join(script_dir, "runs/cifar10_experiment")
    writer = SummaryWriter(writer_dir)  # 日志保存在 runs/cifar10_experiment 文件夹中

    # 训练模型
    train_model(net, trainloader, criterion, optimizer, device, writer, epochs=19)

    # 测试模型
    test_model(net, testloader, device, writer, epoch=19)  # 测试最后一个epoch的结果

    # 保存模型
    model_path = os.path.join(script_dir, "model/cifar10_cnn.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(net.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    writer.close()  # 关闭 TensorBoard 日志记录器

    # 创建测试图像
    create_test_images(num_images=10)

    print("\n要使用TensorBoard查看训练过程，可以在命令行运行： tensorboard --logdir=runs")

if __name__ == '__main__':
    main()