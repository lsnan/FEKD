import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet50Student(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50Student, self).__init__()
        # 使用预训练的 ResNet50 作为基础模型
        self.resnet = resnet50(pretrained=True)

        # 移除 ResNet50 的最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # 自定义分类头，调整通道数
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)

        # 上采样到 (512, 512)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        # 前向传播
        x = self.resnet(x)  # ResNet50 的输出是特征图
        x = self.conv1(x)  # 卷积层降维
        x = self.conv2(x)
        x = self.bn(x)  # 进行批归一化
        x = self.upsample(x)  # 上采样到所需尺寸 (512, 512)
        return x

# 示例用法
if __name__ == "__main__":
    model = ResNet50Student(num_classes=8)
    x = torch.randn(1, 3, 224, 224)  # 假设输入尺寸为 (1, 3, 224, 224)
    output = model(x)
    print(output.shape)