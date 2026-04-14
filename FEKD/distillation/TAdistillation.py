import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import MyData
from studentResNet50 import ResNet50Student
#from models.birefnet import BiRefNet
from student import ResNet101Student
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

from EffectNet_student import EfficientNetB2Student

# 加载教师网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_model = ResNet101Student()
teacher_weights_path = '/home/user/桌面/BiRefNet-main_1/ckpt/COD/student_ResNet101_loss_V3-epoch_50.pth'
teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
teacher_model.to(device)
teacher_model.eval()

# 定义学生网络
student_model = ResNet50Student(num_classes=8)

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

    # 打印当前学习率
    current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    print(f"当前学习率: {current_lr}")

    return total_loss

def configure_optimizers(self):
    # 优化器
    optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)
    # 动态学习率调度器：CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 这里T_max是调整的步数，可以根据需要调整

    return {
        'optimizer': optimizer,
        'lr_scheduler': scheduler
    }

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
        teacher_outputs = (teacher_outputs[1] - teacher_outputs[1].mean()) / teacher_outputs[1].std()

        student_outputs = self.student_model(images)

        loss = combined_loss(student_outputs, teacher_outputs)
        # 打印当前学习率
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"当前学习率: {current_lr}")

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)

# 设置训练过程
distillation_loss = DistillationLoss()
trainer = DistillationTrainer(teacher_model, student_model, distillation_loss, learning_rate=1e-3)

# 获取数据加载器'COD': '+'.join(['COD10K-v3', 'CAMO'][:]),
train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=4)

pl_trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1)
pl_trainer.fit(trainer, train_loader, val_loader)

# 在训练完成后保存学生模型权重
student_state_dict = trainer.student_model.state_dict()
torch.save(student_state_dict, 'lightning_logs/TA-ResNet50_V2.pth')
