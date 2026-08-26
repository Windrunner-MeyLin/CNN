# 导入所需的库
import os  # 操作系统接口，用于文件路径操作
# 设置环境变量解决 OpenMP 运行时冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch  # PyTorch深度学习框架，用于构建和训练神经网络
import torch.nn as nn  # PyTorch的神经网络模块，包含各种层和损失函数
from torch.utils.data import Dataset, DataLoader  # 数据集处理和数据加载工具
from torchvision import transforms, models  # 图像预处理工具和预训练模型
import cv2  # OpenCV库，用于图像读取和处理

# 定义一个卷积神经网络（CNN）模型，用于手写数字识别
class CNN(nn.Module):
    def __init__(self):
        """
        初始化CNN网络结构
        包含两个卷积层、池化层、全连接层和Dropout层
        """
        super(CNN, self).__init__()
        
        # 第一个卷积层：输入1通道（灰度图），输出32通道，使用3x3卷积核，边缘填充1像素
        # 作用：提取图像的低级特征（如边缘、线条）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 第二个卷积层：输入32通道，输出64通道，同样使用3x3卷积核
        # 作用：提取更复杂的特征（如形状、纹理）
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 最大池化层：窗口大小2x2，步长2
        # 作用：降低特征图尺寸，减少计算量，同时保留主要特征
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一个全连接层：输入尺寸64*7*7（经过两次池化后的特征图大小），输出128个特征
        # 作用：将卷积层提取的特征进行组合，准备分类
        self.fc1 = nn.Linear(64*7*7, 128)
        
        # 第二个全连接层：输入128个特征，输出10个类别（对应数字0-9）
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层：随机丢弃50%的神经元，防止模型过拟合（仅在训练时生效）
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        定义数据的前向传播过程
        x: 输入图像数据（形状：[batch_size, 1, 28, 28]）
        返回值：10个类别的预测概率
        """
        # 卷积层1 -> ReLU激活 -> 池化（输出形状：[batch_size, 32, 14, 14]）
        x = self.pool(torch.relu(self.conv1(x)))
        
        # 卷积层2 -> ReLU激活 -> 池化（输出形状：[batch_size, 64, 7, 7]）
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 将特征图展平为一维向量（形状：[batch_size, 64*7*7=3136]）
        x = x.view(-1, 64*7*7)
        
        # 全连接层1 -> ReLU激活
        x = torch.relu(self.fc1(x))
        
        # 应用Dropout（训练时随机丢弃部分神经元，测试时不丢弃）
        x = self.dropout(x)
        
        # 全连接层2（输出形状：[batch_size, 10]）
        x = self.fc2(x)
        
        return x

# 自定义数据集类，用于加载和处理图像数据
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图像文件夹路径
        transform: 图像预处理流程
        """
        self.img_dir = img_dir
        # 获取所有.png/.jpg/.jpeg结尾的文件名
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        """返回数据集的样本数量"""
        return len(self.img_names)
    
    def __getitem__(self, idx):
        """加载并返回索引为idx的样本（图像和标签）"""
        # 拼接完整文件路径
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        
        # 使用OpenCV读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"无法读取图像：{img_path}")
        
        # 将BGR格式转换为RGB格式（OpenCV默认读取为BGR）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 提取文件名的数字部分
        label = int(os.path.splitext(self.img_names[idx])[0])
        
        # 应用预处理流程（如有）
        if self.transform:
            image = self.transform(image)
            
        return image, label

def deal_data(data_dir, batch_size=1):
    """
    构建数据预处理流程并创建数据加载器
    data_dir: 数据文件夹路径
    batch_size: 每次加载的样本数量
    """
    # 定义图像预处理步骤（顺序严格按照MNIST规范设计）
    transform = transforms.Compose([
        transforms.ToPILImage(),                    # 1. 将numpy数组转换为PIL图像格式
        transforms.Resize((28, 28)),               # 2. 调整图像大小为28x28像素
        transforms.Grayscale(num_output_channels=1),# 3. 转换为单通道灰度图
        transforms.RandomInvert(p=1.0),             # 4. 颜色反转（黑底白字转为白底黑字）
        transforms.ToTensor(),                      # 5. 转换为Tensor并自动归一化到[0.0, 1.0]
        transforms.Normalize((0.1307,), (0.3081,))  # 6. 标准化（使用MNIST官方统计参数）
    ])
    
    # 创建数据集实例
    dataset = CustomImageDataset(
        img_dir=data_dir,
        transform=transform
    )
    
    # 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,       # 测试时不需要打乱数据顺序
        num_workers=0        # 设置为0避免多进程路径编码问题
    )
    
    return loader

def test(model, device, test_loader):
    """评估模型在测试集上的准确率"""
    model.eval()  # 将模型切换到评估模式（关闭Dropout和BatchNorm的随机性）
    
    if not model or not test_loader:
        raise RuntimeError("需要先加载模型和数据集！")
    
    correct = 0  # 记录正确预测的数量
    
    # 禁用梯度计算（节省内存，加快计算速度）
    with torch.no_grad():
        for data, target in test_loader:
            # 将数据移动到设备（GPU/CPU）
            data, target = data.to(device), target.to(device)
            
            # 前向传播，得到预测结果
            output = model(data)
            
            # 获取预测类别（输出中最大值的索引）
            pred = output.argmax(dim=1, keepdim=True)
            
            # 累加正确预测数
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 打印每个样本的预测结果
            print(f"真实标签：{target.item()}，预测结果：{pred.item()}")
    
    # 计算准确率（正确数 / 总样本数 * 100）
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return accuracy

def main():
    # 程序主入口
    # 选择设备：优先使用GPU（如果可用），否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型文件路径（使用相对路径避免中文路径问题）
    model_path = "./model/mnist_cnn.pth"
    
    # 创建CNN实例，并将模型移动到设备
    model = CNN().to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_path))
    
    # 测试图像文件夹路径（使用相对路径避免中文路径问题）
    image_path = "./data/numbersbyHAND"
    
    # 创建测试数据加载器
    test_loader = deal_data(image_path)
    
    # 测试模型
    accuracy = test(model, device, test_loader)
    
    # 输出最终准确率
    print(f'模型准确率：{accuracy:.1f}%')

if __name__ == "__main__":
    main()