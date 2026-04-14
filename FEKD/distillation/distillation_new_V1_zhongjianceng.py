import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import MyData
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.birefnet import BiRefNet
# from student import ResNet101Student
from distillation.student import ResNet101Student#lsn

# 定义combined_loss函数
def combined_loss(student_output, teacher_output, temperature=2.0, alpha=0.5, beta=0.5):
    #lsn--------------------------------------------------
    print(">>> type(teacher_output):", type(teacher_output))
    if isinstance(teacher_output, (list, tuple)):
        print(">>> teacher_output length:", len(teacher_output))
        for i, t in enumerate(teacher_output):
            print(f"  [{i}] type: {type(t)}, shape: {getattr(t, 'shape', None)}")
#-------------------------------------------------------------------------
    # 温度缩放
    student_output_scaled = student_output / temperature
    teacher_output_scaled = teacher_output / temperature

    # MSE 损失
    mse_loss = F.mse_loss(student_output_scaled, teacher_output_scaled)

    # L1 损失
    l1_loss = F.l1_loss(student_output_scaled, teacher_output_scaled)

    # 总损失
    total_loss = alpha * mse_loss + beta * l1_loss
    return total_loss

# 特征对齐损失
def feature_alignment_loss(student_features, teacher_features):
    # 在多个中间层进行对齐，计算 MSE 损失
    alignment_loss = 0.0
    for student_feature, teacher_feature in zip(student_features, teacher_features):
        alignment_loss += F.mse_loss(student_feature, teacher_feature)
    return alignment_loss

# 训练过程
class DistillationTrainer(pl.LightningModule):
    def __init__(self, teacher_model, student_model, learning_rate=1e-3):
        super(DistillationTrainer, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        images, labels, *_ = batch

        # a = self.teacher_model(images)
        # i = 1
        # 获取教师网络的中间层输出和最终输出
        with torch.no_grad():
            teacher_output, teacher_features = self.teacher_model(images)
#lsn--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 如果 teacher_output 是 list，只取最后一个（即最终输出）
        if isinstance(teacher_output, (list, tuple)):
            teacher_output = teacher_output[-1]
            #-----------------------------------------------------------
        # 获取学生网络的中间层输出和最终输出
        student_features, student_output = self.student_model(images)

        # 计算中间层特征对齐损失
        feature_loss = feature_alignment_loss(student_features, teacher_features)

        # 计算最终输出的 combined_loss
        output_loss = combined_loss(student_output, teacher_output)

        # 总损失：中间层损失 + 输出损失
        total_loss = feature_loss + output_loss

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

# 数据加载器
def get_data_loaders(batch_size, num_workers):
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3_1', image_size=512, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3_1', image_size=512, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# 设置训练过程
trainer_module = DistillationTrainer(teacher_model, student_model, learning_rate=1e-3)

# 获取数据加载器
train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=4)

# 使用 PyTorch Lightning 进行训练
pl_trainer = pl.Trainer(max_epochs=1, accelerator='gpu', devices=1)
pl_trainer.fit(trainer_module, train_loader, val_loader)

# 在训练完成后保存学生模型权重
student_state_dict = trainer_module.student_model.state_dict()
# torch.save(student_state_dict, 'lightning_logs/student_ResNet101_new_loss_V6.pth')
torch.save(student_state_dict, 'lightning_logs/student_ResNet101_new_loss_Vn.pth')#lsn
