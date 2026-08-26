import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

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

class CustomImageDataset(Dataset):
    """自定义数据集类，用于加载批量图像"""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        # 获取所有图像文件（只处理png格式）
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                         if f.endswith('.png')]
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 定义CIFAR-10类别标签
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 从文件名提取标签（文件名格式为 "[标签][唯一标识].png"）
        filename = os.path.basename(img_path)
        # 去掉 .png 后缀，然后提取第一个 [] 中的类别名
        name_without_ext = filename[:-4]  # 去掉 .png
        # 格式为 [类别][数字]，提取第一个 [] 中的内容
        class_name = name_without_ext.split('][')[0][1:]  # [cat][0] -> cat
        label = classes.index(class_name)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def predict_batch_images(model_path, img_dir, batch_size=4):
    """批量预测图像并计算top-1准确率"""
    # 定义CIFAR-10类别标签
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 图像预处理流程
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建数据集和数据加载器
    dataset = CustomImageDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 统计正确预测数量
    correct = 0
    total = 0
    
    print("=== 批量预测结果 ===")
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # 统计正确预测
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # 打印每个图像的预测结果
            for i in range(len(paths)):
                filename = os.path.basename(paths[i])
                actual_label = classes[labels[i].item()]
                predicted_label = classes[predicted[i].item()]
                status = "OK" if predicted[i] == labels[i] else "FAIL"
                print(f"{filename}: 真实={actual_label}, 预测={predicted_label} {status}")
    
    # 计算top-1准确率
    accuracy = 100 * correct / total
    print(f"\n=== Top-1 准确率: {correct}/{total} = {accuracy:.2f}% ===")
    
    return accuracy

def main():
    # 模型文件路径
    model_path = "./model/cifar10_cnn.pth"
    
    # 测试图像文件夹路径
    img_dir = "./test_images/"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行CNNv3.py训练模型")
        return
    
    if not os.path.exists(img_dir):
        print(f"错误：图像文件夹不存在: {img_dir}")
        print("请创建测试图像文件夹并放入图像")
        return
    
    # 执行批量预测
    predict_batch_images(model_path, img_dir)

if __name__ == "__main__":
    main()