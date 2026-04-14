import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from selfStudent import SimplifiedBiRefNet
from models.birefnet import BiRefNet
from dataset import MyData
from torch.utils.data import DataLoader

# 使用均方误差
class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs):
        # 直接计算学生和教师网络输出的特征图的 MSE 损失
        loss = self.mse_loss(student_outputs, teacher_outputs)
        return loss

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.5, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.temperature = temperature
        # self.alpha = alpha
        # self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        # # 软目标：教师网络输出
        # teacher_probs = F.softmax(teacher_outputs[-1] / self.temperature, dim=1)
        # # 学生网络 log 软目标
        # student_log_probs = F.log_softmax(student_outputs / self.temperature, dim=1)
        #
        # # 知识蒸馏损失：KL散度
        # loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)
        # # 交叉熵损失
        # loss_ce = self.ce_loss(student_outputs, labels)
        #
        # # 结合两者损失
        # return self.alpha * loss_ce + (1 - self.alpha) * loss_kd

        # 只使用最后一层的输出
        teacher_output_last = teacher_outputs[-1]
        student_output_last = student_outputs[0][-1][-1]

        # 计算 MSE 损失
        mse_loss = self.mse_loss(student_output_last, teacher_output_last)
        a = 1

        # 返回均方误差损失
        return mse_loss


# 自蒸馏训练函数
def train_self_distillation(teacher_model, student_model, dataloader, distillation_loss, optimizer, device):
    teacher_model.eval()  # 固定教师模型
    student_model.train()  # 训练学生模型

    running_loss = 0.0
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)

        # 教师网络的输出
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # 学生网络的输出
        student_outputs = student_model(images)

        # 计算蒸馏损失
        loss = distillation_loss(student_outputs, teacher_outputs, labels)

        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def get_data_loaders(batch_size, num_workers):
    train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载教师模型
teacher_model = BiRefNet()
teacher_weights_path = '/home/user/桌面/BiRefNet-main/ckpt/BSL/BiRefNet-COD-epoch_125.pth'
teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
teacher_model.to(device)

# 定义学生模型
student_model = BiRefNet().to(device)

# 定义数据加载器
# train_dataset = MyData('path_to_train_data', image_size=512, is_train=True)
train_loader = get_data_loaders(batch_size=2, num_workers=4)

# 定义损失函数、优化器
distillation_loss = DistillationLoss(temperature=1.5, alpha=0.7)
optimizer = optim.AdamW(student_model.parameters(), lr=1e-4)  # 这里先使用AdamW

# 训练自蒸馏模型
epochs = 50
for epoch in range(epochs):
    epoch_loss = train_self_distillation(teacher_model, student_model, train_loader, distillation_loss, optimizer, device)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

# 保存训练好的学生模型
torch.save(student_model.state_dict(), 'lightning_logs/selfKD_T1-5_a-7_V1.pth.pth')