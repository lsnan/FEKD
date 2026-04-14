import torch
import torch.nn as nn
from torchvision.models import resnet101
from models.modules.gated_freq_fusion_block import GatedFreqFusionBlock  # 频域模块

class ResNet101Student(nn.Module):
    def __init__(self, num_classes=8,use_freq=False):
        super(ResNet101Student, self).__init__()
        self.use_freq = use_freq
        # 使用预训练 ResNet101
        base_resnet = resnet101(weights='IMAGENET1K_V1')

        # 分阶段拆分
        self.stage1 = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.stage2 = base_resnet.layer1   # 输出通道 256
        self.stage3 = base_resnet.layer2   # 输出通道 512
        self.stage4 = base_resnet.layer3   # 输出通道 1024
        self.stage5 = base_resnet.layer4   # 输出通道 2048

        # 频域增强模块
        if self.use_freq:
            self.freq_block3 = GatedFreqFusionBlock(512, reduction=4, gain=0.1)
            # self.freq_block3 = None
            self.freq_block4 = GatedFreqFusionBlock(1024, reduction=4, gain=0.1)
        else:
            self.freq_block3 = None
            self.freq_block4 = None
        # 分类头
        self.conv1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(num_classes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

    def forward(self, x):
        # backbone
        x = self.stage1(x)
        x = self.stage2(x)

#####开启GFFB-----------------------
        # x3 = self.stage3(x)
        # # x3_f = x3
        # x3_f = self.freq_block3(x3)
        #
        # x4 = self.stage4(x3_f)
        # x4_f = self.freq_block4(x4)
###----------------------------------------
# ###若关闭GFFB，则使用下面代码--------------------
        x3 = self.stage3(x)

        # ---- 是否开启频域增强 ----
        if self.use_freq:
            x3_f = self.freq_block3(x3)
        else:
            x3_f = x3  # 关闭时直接用原特征

        x4 = self.stage4(x3_f)

        if self.use_freq:
            x4_f = self.freq_block4(x4)
        else:
            x4_f = x4
# #####--------------------------------------
        x5 = self.stage5(x4_f)

        # 分类头
        out = self.conv1(x5)
        out = self.conv2(out)
        out = self.bn(out)
        out_final = self.upsample(out)

        # 返回中间特征列表，用于蒸馏
        feats = [x3_f, x4_f]
        return out_final, feats


# 测试
if __name__ == "__main__":
    model = ResNet101Student(num_classes=8)
    x = torch.randn(1, 3, 224, 224)
    out, feats = model(x)
    print("预测结果:", out.shape)
    print("中间特征:", [f.shape for f in feats])
