
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import MyData
from models.birefnet import BiRefNet
from student import MobileNetV2Student

# 学生网络
class MobileNetV2Student(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileNetV2Student, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, num_classes)
        )
        self.conv1 = nn.Conv2d(self.mobilenet.last_channel, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.mobilenet.features(x)
        x = self.conv1(x)  # 卷积层降维
        x = self.conv2(x)
        x = self.bn(x)
        x = self.upsample(x)
        return x

# 加载教师网络
teacher_model = BiRefNet()
teacher_model.load_state_dict(torch.load('/home/user/桌面/BiRefNet-main/ckpt/BSL/epoch_115.pth'))
teacher_model.eval()

# 定义学生网络
student_model = MobileNetV2Student(num_classes=8)

# 蒸馏损失函数
class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        if isinstance(student_outputs, list):
            student_outputs = student_outputs[0]
        if isinstance(teacher_outputs, list):
            teacher_outputs = teacher_outputs[3]

        # 计算软目标损失
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)

        # 计算知识蒸馏损失
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

        # 计算交叉熵损失
        loss_ce = self.ce_loss(student_outputs, labels)

        # 结合两部分损失
        return self.alpha * loss_ce + (1 - self.alpha) * loss_kd

def combined_distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5, temperature=1.0):
    mse_loss = F.mse_loss(student_outputs, teacher_outputs)
    kl_loss = F.kl_div(F.log_softmax(student_outputs / temperature, dim=1),
                       F.softmax(teacher_outputs / temperature, dim=1),
                       reduction='batchmean') * (temperature ** 2)
    return alpha * mse_loss + (1 - alpha) * kl_loss


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
        #loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
        loss = combined_distillation_loss(student_outputs, teacher_outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)

# 设置训练过程
distillation_loss = DistillationLoss(temperature=1.5, alpha=0.7)
trainer = DistillationTrainer(teacher_model, student_model, distillation_loss, learning_rate=1e-4)

# 获取数据加载器
train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=4)

pl_trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1)
pl_trainer.fit(trainer, train_loader, val_loader)

# 在训练完成后保存学生模型权重
student_state_dict = trainer.student_model.state_dict()
torch.save(student_state_dict, 'lightning_logs/11111111.pth')

