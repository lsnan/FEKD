import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class EfficientNetB2Student(nn.Module):
    def __init__(self, num_classes=8):
        super(EfficientNetB2Student, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b2')
        self.conv1 = nn.Conv2d(1408, num_classes, kernel_size=1)  # 1408 是 EfficientNet-B2 的最后一个卷积层的输出通道数
        self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.conv1(x)  # 卷积层降维
        x = self.conv2(x)
        x = self.bn(x)
        x = self.upsample(x)
        return x
