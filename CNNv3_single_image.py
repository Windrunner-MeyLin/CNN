import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from PIL import Image

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        # 第一组卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 第二组卷积
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # 第三组卷积
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_image(image_path):
    """加载并预处理单张图像"""
    # 图像预处理流程（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),           # 调整图像大小为32x32
        transforms.ToTensor(),                  # 转换为Tensor并归一化到[0.0, 1.0]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 标准化
    ])
    
    # 使用PIL读取图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image

def predict_single_image(model_path, image_path):
    """预测单张图像的类别"""
    # 定义CIFAR-10类别标签
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = CNN().to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载并预处理图像
    image = load_image(image_path).to(device)
    
    # 执行预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)
    
    # 获取预测结果和置信度
    predicted_class = classes[predicted.item()]
    confidence = probabilities[0][predicted.item()].item() * 100
    
    print(f"图像: {image_path}")
    print(f"预测类别: {predicted_class}")
    print(f"置信度: {confidence:.2f}%")
    
    return predicted_class, confidence

def main():
    # 模型文件路径
    model_path = "./model/cifar10_cnn.pth"
    
    # 测试图像路径（用户需要替换为自己的图像路径）
    image_path = "./test_image.png"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行CNNv3.py训练模型")
        return
    
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在: {image_path}")
        print("请将测试图像放在正确路径或修改image_path变量")
        return
    
    # 执行预测
    predict_single_image(model_path, image_path)

if __name__ == "__main__":
    main()