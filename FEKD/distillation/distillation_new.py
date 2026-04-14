import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from dataset import MyData
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.birefnet_old import BiRefNet
# from student import ResNet101Student
from distillation.student import ResNet101Student#lsn
from torch.utils.data import ConcatDataset
from models.modules.mini_freq_attn import MiniFreqAttn   ##lsnmini
from pytorch_lightning.callbacks import ModelCheckpoint
# 定义combined_loss函数
def combined_loss(student_output, teacher_output, temperature=2.0, alpha=0.5, beta=0.5):
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

####lsnmini教师频域------------------------------------------
# ==================== 频域蒸馏损失 ====================
class FreqAwareLoss(nn.Module):
    def __init__(self, beta=0.3):
        super(FreqAwareLoss, self).__init__()
        self.beta = beta

    def forward(self, student_feat, teacher_feat):
        # 对齐通道数
        if student_feat.shape[1] != teacher_feat.shape[1]:
            student_feat = F.interpolate(student_feat, size=teacher_feat.shape[2:], mode='bilinear',
                                         align_corners=False)
            student_feat = student_feat.mean(dim=1, keepdim=True)  # 临时通道压缩
        else:
            student_feat_aligned = student_feat

        # 计算空间蒸馏损失
        spatial_loss = F.mse_loss(student_feat_aligned, teacher_feat.detach())

        # ---- 频域损失 ----
        S_f = torch.fft.fft2(student_feat)
        T_f = torch.fft.fft2(teacher_feat.detach())

        S_f = torch.abs(S_f)
        T_f = torch.abs(T_f)

        # 防止能量差异过大，归一化
        S_f = S_f / (S_f.mean(dim=(2,3), keepdim=True) + 1e-8)
        T_f = T_f / (T_f.mean(dim=(2,3), keepdim=True) + 1e-8)

        if S_f.shape[1] != T_f.shape[1]:
            conv = torch.nn.Conv2d(S_f.shape[1], T_f.shape[1], kernel_size=1).to(S_f.device)
            S_f_aligned = conv(S_f)
        else:
            S_f_aligned = S_f

        freq_loss = F.mse_loss(S_f_aligned, T_f)

        total_loss = 0.5 * spatial_loss + 0.5 * self.beta * freq_loss
        return total_loss
#####-----------------------------------------------------------------
###lsnmini教师频域 ==================== 通道对齐模块 ====================
class ChannelAlign(nn.Module):
    """用于将教师特征映射到与学生特征相同通道数"""
    def __init__(self, in_channels, out_channels):
        super(ChannelAlign, self).__init__()
        self.align = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.align(x)))
###----------------------------------------------------------

# 特征对齐损失
def feature_alignment_loss(student_features, teacher_features):
    # 在多个中间层进行对齐，计算 MSE 损失
    alignment_loss = 0.0
    for student_feature, teacher_feature in zip(student_features, teacher_features):
        alignment_loss += F.mse_loss(student_feature, teacher_feature)
    return alignment_loss

# 训练过程
class DistillationTrainer(pl.LightningModule):
    def __init__(self, teacher_model, student_model, learning_rate=1e-3,use_align=False):
        self.use_align = use_align

        super(DistillationTrainer, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

    #lsnmini教师频域------------------------
        # 对齐教师特征通道到学生通道
        if self.use_align:
            self.align_32 = ChannelAlign(in_channels=1, out_channels=1024)
            self.align_64 = ChannelAlign(in_channels=1, out_channels=512)
            self.align_512 = ChannelAlign(in_channels=1, out_channels=256)
        else:
            self.align_32 = None
            self.align_64 = None
            self.align_512 = None

        ####-----------------------------------------
        self.learning_rate = learning_rate
#################保存多轮权重文件-----------------------------------------
    def on_train_epoch_end(self):
        """每 10 轮保存一次 .pth 权重文件，并在最后一轮再保存一次"""
        epoch = self.current_epoch
        # max_epochs = self.trainer.max_epochs
        # 保存路径
        save_dir = "weights_pth"
        os.makedirs(save_dir, exist_ok=True)

        # 每 10 轮保存一次
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(save_dir, f"student_epoch_{epoch + 1}.pth")
            torch.save(self.student_model.state_dict(), save_path)
            print(f"✔ Saved checkpoint: {save_path}")
        # -------- 2. 最后 20 轮全部保存 --------
        # if epoch + 1 > max_epochs - 20:
        #     save_path = os.path.join(save_dir, f"student_epoch_{epoch + 1}.pth")
        #     torch.save(self.student_model.state_dict(), save_path)
        #     print(f"✔ Saved checkpoint (last 20 epochs): {save_path}")
    def on_train_end(self):
        """训练结束后保存最终权重"""
        save_dir = "weights_pth"
        os.makedirs(save_dir, exist_ok=True)

        final_path = os.path.join(save_dir, "student_final.pth")
        torch.save(self.student_model.state_dict(), final_path)
        print(f"✔ Saved final model: {final_path}")
###########---------------------------------------------
    def forward(self, x):
        return self.student_model(x)

    # def training_step(self, batch, batch_idx):
    #     images, labels, *_ = batch
    #
    #     # 获取教师网络的输出
    #     with torch.no_grad():
    #         teacher_outputs = self.teacher_model(images)
    #
    #     # 标准化教师网络的输出（可以根据需求进行）
    #     teacher_outputs = (teacher_outputs[3] - teacher_outputs[3].mean()) / teacher_outputs[3].std()
    #     # teacher_outputs = teacher_outputs[3]
    #
    #     # 获取学生网络的输出
    #     student_outputs = self.student_model(images)
    #
    #     # 计算 combined_loss
    #     loss = combined_loss(student_outputs, teacher_outputs)
    #     # 打印当前学习率
    #     current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
    #     print(f"当前学习率: {current_lr}")
    #
    #     return loss
#####lsnmini教师频域------注释上面training_step函数-----
    def training_step(self, batch, batch_idx):


        def parse_outputs(out):
            """统一解析模型输出格式，返回 (pred, feats)"""
            if isinstance(out, (list, tuple)):
                if len(out) == 2 and isinstance(out[1], (list, tuple)):
                    pred = out[0]
                    feats = list(out[1])
                else:
                    pred = out[0]
                    feats = list(out[1:])
            elif isinstance(out, dict):
                pred = out.get('out', out.get('pred', next(iter(out.values()))))
                feats = []
            else:
                pred = out
                feats = []
            return pred, feats

        images, labels, *_ = batch
        device = images.device

        # ===== 教师输出 =====
        with torch.no_grad():
            teacher_pred, teacher_feats = parse_outputs(self.teacher_model(images))

        # ===== 学生输出 =====
        student_pred, student_feats = parse_outputs(self.student_model(images))

        # 如果教师/学生没有中间特征列表，用 pred 做占位（数量按学生特征数或1）
        if len(teacher_feats) == 0:
            # 如果 student 有中间特征，就复制 teacher_pred 对齐 student 的长度；否则保留单个 pred
            if len(student_feats) > 0:
                teacher_feats = [teacher_pred.detach()] * len(student_feats)
            else:
                teacher_feats = [teacher_pred.detach()]

        if len(student_feats) == 0:
            # 若学生没有中间特征（不常见），用 student_pred 做占位
            student_feats = [student_pred]

        # ===== Sanity print（只在 batch_idx==0 打印，便于调试） =====
        if batch_idx == 0:
            print("\n=== Debug: Forward check ===")
            print("teacher_pred:", tuple(teacher_pred.shape))
            print("student_pred:", tuple(student_pred.shape))
            print("teacher_feats:", [tuple(f.shape) for f in teacher_feats])
            print("student_feats:", [tuple(f.shape) for f in student_feats])
            print("===========================\n")

        # ===== 损失计算（对齐空间尺寸） =====
        freq_loss_fn = FreqAwareLoss(beta=0.3)####消融

        # 对齐并计算主预测损失（下采样 student 到 teacher）
        # 确保 teacher_pred 和 student_pred 是 tensor
        if not isinstance(teacher_pred, torch.Tensor) or not isinstance(student_pred, torch.Tensor):
            raise RuntimeError("teacher_pred or student_pred is not a tensor")

        t_h, t_w = teacher_pred.shape[2], teacher_pred.shape[3]
        s_h, s_w = student_pred.shape[2], student_pred.shape[3]

        if (s_h, s_w) != (t_h, t_w):
            student_pred_scaled = torch.nn.functional.interpolate(
                student_pred, size=(t_h, t_w), mode='bilinear', align_corners=False
            )
        else:
            student_pred_scaled = student_pred

        loss_basic = combined_loss(student_pred_scaled, teacher_pred)
##消融----------------------------------
        # 频域特征损失：逐层对齐空间尺寸再计算
        # ===== 改进版多尺度特征蒸馏 =====
        loss_freq = 0.0
        valid_pairs = 0

        # 教师特征通道对齐（自动匹配不同层）
        for i, (s_feat, t_feat) in enumerate(zip(student_feats, teacher_feats)):
            if not isinstance(s_feat, torch.Tensor) or not isinstance(t_feat, torch.Tensor):
                continue

            # 空间尺寸对齐
            th, tw = t_feat.shape[2], t_feat.shape[3]
            if s_feat.shape[2:] != (th, tw):
                s_feat = F.interpolate(s_feat, size=(th, tw), mode='bilinear', align_corners=False)

            # 根据层索引选择对应对齐器
            # ===== 是否启用ChannelAlign =====
            if self.use_align:
                if i == 0:
                    t_feat = self.align_64(t_feat)
                    weight = 0.6
                elif i == 1:
                    t_feat = self.align_32(t_feat)
                    weight = 0.8
                else:
                    t_feat = self.align_512(t_feat)
                    weight = 0.3
            else:
                # ❗关闭通道对齐时，用最简单规则：直接调整通道数
                if t_feat.shape[1] != s_feat.shape[1]:
                    # 这里简单做均值压缩避免报错（不引入新的学习参数）
                    t_feat = t_feat.mean(dim=1, keepdim=True)
                    t_feat = t_feat.repeat(1, s_feat.shape[1], 1, 1)

                weight = 0.5

            loss_i = freq_loss_fn(s_feat, t_feat)
            loss_freq += weight * loss_i
            valid_pairs += 1

        if valid_pairs > 0:
            loss_freq /= valid_pairs
        else:
            loss_freq = torch.tensor(0.0, device=device)

            if valid_pairs > 0:
                loss_freq = loss_freq / valid_pairs
            else:
                loss_freq = torch.tensor(0.0, device=device)
####消融---------------------

        # 总损失
        loss = loss_basic + 0.5 * loss_freq#####消融
        # loss = loss_basic
        # loss = 1.0 * loss_basic + 0.2 * loss_freq

        # ===== 日志 =====
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}, loss={loss.item():.6f}")
#####消融-----------------------
        print(f"[Epoch {self.current_epoch} | Batch {batch_idx}] "
              f"Loss_total={loss.item():.3f}, Basic={loss_basic.item():.3f}, Freq={loss_freq.item():.3f}")
        # print(f"[Epoch {self.current_epoch} | Batch {batch_idx}] "
        #       f"Loss_total={loss.item():.3f}, Basic={loss_basic.item():.3f}")
        #------------------
        return loss
    #####----------------------------------------------------------------------


    def configure_optimizers(self):
        # 优化器
#########-------------------------------------
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)
        # 动态学习率调度器：CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=25)  # 这里T_max是调整的步数，可以根据需要调整

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)

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
# def get_data_loaders(batch_size, num_workers):
#     train_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#     val_dataset = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#
#     return train_loader, val_loader

def get_data_loaders(batch_size, num_workers):
    # 加载COD10K数据集
    cod_train = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=True)
    cod_val = MyData(datasets='/home/user/lc/datasets/COD/COD10K-v3', image_size=512, is_train=False)

    # 加载CAMO数据集
    camo_train = MyData(datasets='/home/user/lc/datasets/COD/CAMO', image_size=512, is_train=True)
    camo_val = MyData(datasets='/home/user/lc/datasets/COD/CAMO', image_size=512, is_train=False)
 # 合并数据集
    train_dataset = ConcatDataset([cod_train, camo_train])
    val_dataset = ConcatDataset([cod_val, camo_val])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

# 设置训练过程
trainer_module = DistillationTrainer(teacher_model, student_model, learning_rate=1e-3)

# 获取数据加载器student_ResNet101_new_loss_V4.pth
train_loader, val_loader = get_data_loaders(batch_size=2, num_workers=4)

# 使用 PyTorch Lightning 进行训练
pl_trainer = pl.Trainer(max_epochs=70, accelerator='gpu', devices=1)
pl_trainer.fit(trainer_module, train_loader, val_loader)

# 在训练完成后保存学生模型权重
student_state_dict = trainer_module.student_model.state_dict()
# torch.save(student_state_dict, 'lightning_logs/student_ResNet101_combine_loss_V4.pth')
torch.save(student_state_dict, 'lightning_logs/student_fre_OnlyF_70.pth')#lsn


