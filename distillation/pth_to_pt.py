import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from student import ResNet101Student

model = ResNet101Student(num_classes=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('/home/user/桌面/BiRefNet-main/distillation/lightning_logs/student_ResNet101_loss_V3.pth'))
model.to(device)
model.eval()

# 定义输入张量的示例（与实际输入形状一致）
example_input = torch.randn(1, 3, 224, 224).to(device)  # 示例输入

# 转换为 TorchScript 模型
traced_model = torch.jit.trace(model, example_input)

# 保存为 .pt 文件
traced_model.save("/home/user/桌面/BiRefNet-main_1/distillation/student.pt")



