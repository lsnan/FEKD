import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import MyData
from models.birefnet import BiRefNet
# from student import ResNet101Student
from distillation.student import ResNet101Student#lsn
import torchvision
# from student import MobileNetV2Student
# from EffectNet_student import EfficientNetB2Student

# 学生网络
# class MobileNetV2Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(MobileNetV2Student, self).__init__()
#         self.mobilenet = mobilenet_v2(pretrained=True)
#         self.mobilenet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.mobilenet.last_channel, num_classes)
#         )
#         self.conv = nn.Conv2d(num_classes, 1, kernel_size=1)  # 假设最后的输出通道数是1
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
#
    # def forward(self, x):
    #     x = self.mobilenet.features(x)
    #     x = x.mean([2, 3])  # 全局平均池化
    #     x = self.mobilenet.classifier(x)
    #     x = x.view(x.size(0), -1, 1, 1)  # 重新调整形状以匹配卷积层的输入
    #     x = self.conv(x)
    #     x = self.upsample(x)
    #     return x

# class MobileNetV2Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(MobileNetV2Student, self).__init__()
#         self.mobilenet = mobilenet_v2(pretrained=True)
#         self.mobilenet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(self.mobilenet.last_channel, num_classes)
#         )
#         self.conv1 = nn.Conv2d(self.mobilenet.last_channel, num_classes, kernel_size=1)
#         self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
#         self.bn = nn.BatchNorm2d(1)
#
#     def forward(self, x):
#         x = self.mobilenet.features(x)
#         x = self.conv1(x)  # 卷积层降维
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.upsample(x)
#         return x

# 加载教师网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = BiRefNet()
# teacher_weights_path = '/home/user/桌面/BiRefNet-main/ckpt/BSL/BiRefNet-COD-epoch_125.pth'
teacher_weights_path = '/media/user/b7c077e3-1e3a-4d1b-b7cc-5a47704b176e/lsn/code/BiRefNet-main_1/ckpt/BSL/BiRefNet-COD-epoch_125.pth'
teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
teacher_model.to(device)
teacher_model.eval()

# 定义学生网络
student_model = ResNet101Student(num_classes=8)
# 使用EffectNetB3
# student_model = EfficientNetB2Student(num_classes=8)

# 蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()
        # self.temperature = temperature
        # self.alpha = alpha
        # self.ce_loss = nn.CrossEntropyLoss()
        #self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        if isinstance(student_outputs, list):
            student_outputs = student_outputs[0]
        if isinstance(teacher_outputs, list):
            teacher_outputs = teacher_outputs[3]
        # 使用mseloss
        mse_loss = combined_loss(student_outputs, teacher_outputs)

        return mse_loss

        # a = 1
        # # 计算软目标损失
        # teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        # student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        #
        # # 计算知识蒸馏损失
        # loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        #
        # # 计算交叉熵损失
        # loss_ce = self.ce_loss(student_outputs, labels)
        #
        # # 结合两部分损失
        # return self.alpha * loss_ce + (1 - self.alpha) * loss_kd


def combined_loss(student_output, teacher_output, temperature=2.0, alpha=0.5,
                  beta=0.5):
    # 温度缩放
    student_output_scaled = student_output / temperature
    teacher_output_scaled = teacher_output / temperature

    # 基本 MSE 损失
    mse_loss = F.mse_loss(student_output_scaled, teacher_output_scaled)

    # L1 损失
    l1_loss = F.l1_loss(student_output_scaled, teacher_output_scaled)

    # 总损失
    total_loss = alpha * mse_loss + beta * l1_loss
    return total_loss


# 数据加载器
def get_data_loaders(batch_size, num_workers):
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def preprocess_data(images):
    # 例如，对图像进行归一化处理
    images = (images - images.min()) / (images.max() - images.min())
    return images
# 训练过程
class DistillationTrainer(pl.LightningModule):
    def __init__(self, teacher_model, student_model, distillation_loss, learning_rate=1e-3):
        super(DistillationTrainer, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_loss = distillation_loss
        self.learning_rate = learning_rate

    def forward(self, x):
        output = self.some_model(x)
        return output

    def training_step(self, batch, batch_idx):
        images, labels, *_ = batch  # 由于数据集返回三个内容 image， label， class_label 类别标签。
        with torch.no_grad():                                         # label_paths[index] 标签的路径。
            teacher_outputs = self.teacher_model(images)
        # 标准化教师模型输出
        teacher_outputs = (teacher_outputs[3] - teacher_outputs[3].mean()) / teacher_outputs[3].std()

        student_outputs = self.student_model(images)

        loss = combined_loss(student_outputs, teacher_outputs)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)

# 设置训练过程
distillation_loss = DistillationLoss()
trainer = DistillationTrainer(teacher_model, student_model, distillation_loss, learning_rate=1e-3)

# 获取数据加载器'COD': '+'.join(['COD10K-v3', 'CAMO'][:]),
train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=4)

pl_trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=1)
pl_trainer.fit(trainer, train_loader, val_loader)

# 在训练完成后保存学生模型权重
student_state_dict = trainer.student_model.state_dict()
torch.save(student_state_dict, 'lightning_logs/student_ResNet101_loss_Vn.pth')
