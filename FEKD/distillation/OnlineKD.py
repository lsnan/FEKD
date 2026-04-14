import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import MyData
from student import ResNet101Student
from models.birefnet import BiRefNet

# 蒸馏损失函数，包含KL散度与交叉熵损失
class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        # 计算教师和学生的损失
        teacher_probs = F.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        # 计算KL散度损失
        loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        # 计算学生的交叉熵损失
        loss_ce = self.ce_loss(student_outputs, labels)
        # 结合两部分损失
        return self.alpha * loss_ce + (1 - self.alpha) * loss_kd

# 定义在线蒸馏的训练器
class OnlineDistillationTrainer(pl.LightningModule):
    def __init__(self, teacher_model, student_model, distillation_loss, learning_rate=1e-3):
        super(OnlineDistillationTrainer, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_loss = distillation_loss
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        images, labels, *_ = batch
        # 教师和学生网络同时前向传播
        teacher_outputs = self.teacher_model(images)
        student_outputs = self.student_model(images)
        # 计算联合损失
        loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            list(self.student_model.parameters()) + list(self.teacher_model.parameters()),
            lr=self.learning_rate
        )

def get_data_loaders(batch_size, num_workers):
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# 加载教师网络并设置为训练模式
teacher_model = BiRefNet(num_classes=8).to('cuda')  # 教师也可以是BiRefNet或其他更大的模型
student_model = ResNet101Student(num_classes=8).to('cuda')  # 使用ResNet101作为学生模型

# 设置蒸馏损失
distillation_loss = DistillationLoss(temperature=1.5, alpha=0.7)

# 在线蒸馏训练器
trainer = OnlineDistillationTrainer(teacher_model, student_model, distillation_loss, learning_rate=4e-4)

# 获取数据加载器
train_loader, val_loader = get_data_loaders(batch_size=12, num_workers=4)

# 训练模型
pl_trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1)
pl_trainer.fit(trainer, train_loader, val_loader)

# 保存学生模型权重
student_state_dict = trainer.student_model.state_dict()
torch.save(student_state_dict, 'lightning_logs/student_ResNet101_online.pth')
