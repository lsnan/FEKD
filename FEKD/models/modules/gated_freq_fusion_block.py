import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFreqFusionBlock(nn.Module):
    """
    Gated Dual-Stream Frequency Fusion Block (推荐稳定版)
    - 同时处理空间与频域分支
    - 在频域中直接调制复数频谱（稳定性高）
    - 使用门控融合控制频域增强强度
    - 残差式输出，便于嵌入到蒸馏学生网络
    """
    def __init__(self, in_channels, reduction=4, gain=0.1):
        super(GatedFreqFusionBlock, self).__init__()
        mid_channels = in_channels // reduction
        self.gain = gain

        # 频域卷积模块（作用于幅度图）
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 门控控制融合强度
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 输出调整层（融合后的特征平衡）
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        spatial = x

        # ======================
        # Step 1: 频域处理
        # ======================
        ffted = torch.fft.fft2(x, norm='ortho')
        amp = torch.log1p(torch.abs(ffted))
        # 标准化增强稳定性
        amp = (amp - amp.mean(dim=[2, 3], keepdim=True)) / (amp.std(dim=[2, 3], keepdim=True) + 1e-5)
        freq_feat = self.freq_conv(amp)

        # 复数域振幅调制
        modulated_fft = ffted * (1 + torch.tanh(freq_feat) * self.gain)
        freq_out = torch.fft.ifft2(modulated_fft, norm='ortho').real

        # ======================
        # Step 2: 门控融合
        # ======================
        fused = torch.cat([spatial, freq_out], dim=1)
        gate = self.gate(fused)
        out = spatial + gate * freq_out  # 自适应控制频域贡献
        out = self.out_conv(out)

        # ======================
        # Step 3: 残差连接输出
        # ======================
        return out + x
