import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet101
from models.modules.mini_freq_attn import MiniFreqAttn  #lsnmini
from models.modules.gated_freq_fusion_block import GatedFreqFusionBlock  # ✅ 新增

class ResNet101Student(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet101Student, self).__init__()
        # 使用预训练的 ResNet101 作为基础模型
        # self.resnet = resnet101(pretrained=True)
        # 移除 ResNet101 的最后一层全连接层
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

###lsnmini教师频域需要注释    【移除 ResNet101 的最后一层全连接层】这两行---------------------------
        ###拆分resnet阶段

        # 使用预训练的 ResNet101 作为基础模型
        base_resnet = resnet101(pretrained=True)

        # 拆分阶段（保留原始层名）
        self.stage1 = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.stage2 = base_resnet.layer1
        self.stage3 = base_resnet.layer2
        self.stage4 = base_resnet.layer3
        self.stage5 = base_resnet.layer4

 ######----------------------------------------

##lsnmini----------------------------------
        # # stage 3 和 stage 4 输出后分别添加频域注意力模块
        # self.freq_attn_stage3 = MiniFreqAttn(in_channels=1024)  # stage3 输出通道为1024
        # self.freq_attn_stage4 = MiniFreqAttn(in_channels=2048)  # stage4 输出通道为2048
        # ✅ 添加频域模块到stage3和stage4
        self.freq_block3 = GatedFreqFusionBlock(1024, reduction=4, gain=0.1)
        self.freq_block4 = GatedFreqFusionBlock(2048, reduction=4, gain=0.1)

        ###----------------------------------------------------

        # 自定义分类头，调整通道数
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)

        # 上采样到 (512, 512)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        self.bn = nn.BatchNorm2d(1)

#     def forward(self, x):
#         # 前向传播
#         x = self.resnet(x)  # ResNet101 的输出是特征图
# # ###lsnmini------------------------------
# #         # stage 3 后添加频域注意力模块
# #         x = self.freq_attn_stage3(x)  # 在 stage 3 后应用频域注意力
# #         # stage 4 后添加频域注意力模块
# #         x = self.freq_attn_stage4(x)  # 在 stage 4 后应用频域注意力
# # #############--------------------------------------
#         x = self.conv1(x)  # 卷积层降维
#         x = self.conv2(x)
#         x = self.bn(x)  # 进行批归一化
#         x = self.upsample(x)  # 上采样到所需尺寸 (512, 512)
#         return x
# ###lsnmini----------------------------------
#     def forward(self, x):
#         # ---- 分阶段提取 ----
#         layers = list(self.resnet.children())
#         x = x
#
#         # 前4个子模块：conv1, bn1, relu, maxpool
#         for i in range(4):
#             x = layers[i](x)
#
#         # layer1
#         x = layers[4](x)
#         # layer2
#         x = layers[5](x)
#         # layer3
#         x = layers[6](x)
#         x = self.freq_block3(x)  # ✅ 对stage3输出进行频域增强
#
#         # layer4
#         x = layers[7](x)
#         x = self.freq_block4(x)  # ✅ 对stage4输出进行频域增强
#
#         # ---- 分类头 ----
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.upsample(x)
#         return x
#     #-------------------------------------------------


#######lsnmini教师频域---------------------------
    def forward(self, x):
        # ---- backbone ----
        x = self.stage1(x)
        x = self.stage2(x)

        x3 = self.stage3(x)
        x3_f = self.freq_block3(x3)

        x4 = self.stage4(x3_f)
        x4_f = self.freq_block4(x4)

        x5 = self.stage5(x4_f)

        # ---- 分类头 ----
        out = self.conv1(x5)
        out = self.conv2(out)
        out = self.bn(out)
        out_final = self.upsample(out)

        # ---- 返回预测结果 + 中间特征 ----
        feats = [x3_f, x4_f]
        return out_final, feats
####------------------------------------


# class ResNet34Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(ResNet34Student, self).__init__()
#         self.resnet = resnet34(pretrained=True)
#         # 移除 ResNet34 的最后一层全连接层
#         self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
#
#         # 自定义分类头
#         self.conv1 = nn.Conv2d(512, num_classes, kernel_size=1)
#         self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
#         self.bn = nn.BatchNorm2d(1)
#
#     def forward(self, x):
#         x = self.resnet(x)  # ResNet34的输出是特征图，不再是向量
#         x = self.conv1(x)  # 卷积层降维
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.upsample(x)  # 上采样到所需尺寸
#         return x

# class MobileNetV2Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(MobileNetV2Student, self).__init__()
#         self.mobilenet = mobilenet_v2(pretrained=True)
#         last_channel = self.mobilenet.last_channel
#         self.mobilenet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, num_classes)
#         )
#         self.conv1 = nn.Conv2d(last_channel, num_classes, kernel_size=1)
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

# class MobileNetV3Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(MobileNetV3Student, self).__init__()
#         self.mobilenet = mobilenet_v3_small(pretrained=True)
#         last_channel = self.mobilenet.classifier[0].in_features
#         self.mobilenet.classifier = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(last_channel, num_classes)
#         )
#         self.conv1 = nn.Conv2d(last_channel, num_classes, kernel_size=1)
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

# 示例用法
if __name__ == "__main__":
    model = ResNet101Student(num_classes=8)
    x = torch.randn(1, 3, 224, 224)  # 假设输入尺寸为 (1, 3, 224, 224)
    output = model(x)
    print(output.shape)





# ###lsnmini--stage4---------------------------------------------------------------
# import torch
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录到路径
# import torch.nn as nn
# from torchvision.models import resnet101
# from models.modules.mini_freq_attn import MiniFreqAttn  # 轻量频域注意力
#
# class ResNet101Student(nn.Module):
#     def __init__(self, num_classes=8):
#         super(ResNet101Student, self).__init__()
#
#         # ---------------------------
#         # 1️⃣ 主干网络：ResNet101（去掉FC层与平均池化层）
#         # ---------------------------
#         resnet = resnet101(pretrained=True)
#         self.stage1 = nn.Sequential(*list(resnet.children())[:5])   # conv1+bn+relu+maxpool+layer1
#         self.stage2 = resnet.layer2
#         self.stage3 = resnet.layer3
#         self.stage4 = resnet.layer4  # 输出通道2048
#         # ---------------------------
#         # 2️⃣ 轻量频域注意力模块（只处理最高层语义特征）
#         # ---------------------------
#
#         self.freq_attn = MiniFreqAttn(in_channels=2048)
#
#
#         # ---------------------------
#         # ---------------------------
#         # 3️⃣ 分类头/解码层
#         # ---------------------------
#         self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
#         self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
#         self.bn = nn.BatchNorm2d(1)
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
#
#     def forward(self, x):
#         # ---------------------------
#         # 提取多层特征
#         # ---------------------------
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)   # [B, 2048, H/32, W/32]
#         print(f"[DEBUG] After stage4: {x.shape}")
#
#         # ---------------------------
#         # 应用轻量频域注意力
#         # ---------------------------
#         x = self.freq_attn(x)
#         print(f"[DEBUG] After freq_attn: {x.shape}")
#
#         # ---------------------------
#         # 分类头
#         # ---------------------------
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.bn(x)
#         x = self.upsample(x)  # 输出 [B, 1, 512, 512]
#         print(f"[DEBUG] Output: {x.shape}")
#         return x
#
# # ---------------------------
# # 测试脚本入口
# # ---------------------------
# if __name__ == "__main__":
#     # 创建模型
#     model = ResNet101Student(num_classes=8)
#
#     # 创建模拟输入
#     x = torch.randn(1, 3, 224, 224)
#
#     # 前向传播
#     output = model(x)
#
#     print(f"[DEBUG] Final output shape: {output.shape}")
#
#
# #####-------------------------------------------------------
#
#
