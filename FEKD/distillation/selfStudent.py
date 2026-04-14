import torch
import torch.nn as nn
import torch.nn.functional as F
from models.birefnet import BiRefNet


class SimplifiedBiRefNet(nn.Module):
    def __init__(self, num_classes=8):
        super(SimplifiedBiRefNet, self).__init__()
        # 使用 BiRefNet 作为基础网络
        self.backbone = BiRefNet()

        # 移除一些复杂模块，例如压缩块和多尺度监督
        self.backbone.squeeze_module = nn.Identity()  # 移除压缩模块
        self.backbone.ms_supervision = False  # 取消多尺度监督

        # 减少解码器复杂性，减少通道数
        channels = [64, 128, 256, 512]  # 减少通道数
        self.backbone.decoder = SimplifiedDecoder(channels)  # 使用简化的解码器

    def forward(self, x):
        # 保持与原模型一致的前向传播逻辑
        features, _ = self.backbone.forward_enc(x)  # 只使用编码器的输出
        scaled_preds = self.backbone.decoder(features)  # 解码特征
        return scaled_preds


class SimplifiedDecoder(nn.Module):
    def __init__(self, channels):
        super(SimplifiedDecoder, self).__init__()
        # 使用 1x1 卷积来减少通道数
        self.reduce_channels_x3 = nn.Conv2d(5760, 768, kernel_size=1)
        self.reduce_channels_x2 = nn.Conv2d(1536, 384, kernel_size=1)  # 减少 x2 的通道数到 384

        # 使用 BiRefNet 的实际通道数 [192, 384, 768, 1536]
        self.decoder_block4 = nn.Conv2d(1536, 768, 3, padding=1)
        self.decoder_block3 = nn.Conv2d(768, 384, 3, padding=1)
        self.decoder_block2 = nn.Conv2d(384, 192, 3, padding=1)
        self.decoder_block1 = nn.Conv2d(192, 1, 3, padding=1)  # 最终输出为1通道

    def forward(self, features):
        # 解包特征
        if len(features) == 4:
            x, x1, x2, x3 = features  # 学生网络有4个特征
        else:
            raise ValueError(f"Expected 4 features, got {len(features)}")

        print(f"x shape: {x.shape}")  # 打印x的形状
        print(f"x1 shape: {x1.shape}")  # 打印x1的形状
        print(f"x2 shape: {x2.shape}")  # 打印x2的形状
        print(f"x3 shape: {x3.shape}")  # 打印x3的形状

        # 使用 1x1 卷积减少 x3 和 x2 的通道数
        x3 = self.reduce_channels_x3(x3)
        x2 = self.reduce_channels_x2(x2)

        # 解码过程
        p3 = F.interpolate(self.decoder_block3(x3), size=x2.shape[2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(self.decoder_block2(p3 + x2), size=x1.shape[2:], mode='bilinear', align_corners=True)
        p1_out = F.interpolate(self.decoder_block1(p2 + x1), size=x.shape[2:], mode='bilinear', align_corners=True)

        return p1_out


if __name__ == "__main__":
    model = SimplifiedBiRefNet(num_classes=8)
    x = torch.randn(1, 3, 512, 512)  # 示例输入
    output = model(x)
    print(output.shape)  # 输出应为 (1, 1, 512, 512)
