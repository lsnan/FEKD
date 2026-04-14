# # mini
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class MiniFreqAttn(nn.Module):
#     """
#     轻量频域注意力（简化版 FGSA）。
#     设计目标：低侵入、少参数、初始化近似恒等、便于调试。
#     使用位置建议：ResNet layer3 输出（B,1024,H/16,W/16）。
#     """
#     def __init__(self, in_channels, freq_kernel=3, init_zero=True):
#         super().__init__()
#         # 1) channel pool -> 2 -> reduce to 1
#         self.reduce = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=1, bias=True),
#             nn.ReLU(inplace=True)
#         )
#         # 2) small conv(s) on amplitude map (single channel)
#         padding = (freq_kernel - 1) // 2
#         self.freq_conv = nn.Sequential(
#             nn.Conv2d(1, 1, kernel_size=freq_kernel, padding=padding, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(1, 1, kernel_size=1, bias=True)
#         )
#         # 3) project single-channel attn back to C channels
#         self.attn_project = nn.Conv2d(1, in_channels, kernel_size=1, bias=True)
#
#         # initialization: allow option to init to near-zero effect
#         if init_zero:
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     # set weights small (or zero) so initial output ~ identity
#                     nn.init.constant_(m.weight, 0.0)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0.0)
#         else:
#             # default kaiming for convs if not init_zero
#             for m in self.modules():
#                 if isinstance(m, nn.Conv2d):
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x):
#         """
#         x: (B, C, H, W)
#         returns: x * (1 + attn_c)  (residual-style modulation)
#         """
#         B, C, H, W = x.shape
#
#         # channel pooling
#         avg = torch.mean(x, dim=1, keepdim=True)    # B,1,H,W
#         maxv, _ = torch.max(x, dim=1, keepdim=True) # B,1,H,W
#         pool = torch.cat([avg, maxv], dim=1)        # B,2,H,W
#         pool = self.reduce(pool)                    # B,1,H,W
#
#         # FFT -> amplitude (magnitude)
#         # note: torch.fft.fft2 returns complex tensor, take abs then log1p to stabilize
#         ffted = torch.fft.fft2(pool, norm='ortho')  # complex B,1,H,W
#         amp = torch.abs(ffted)
#         amp = torch.log1p(amp)                       # compress dynamic range
#
#         # small conv on amp -> single-channel attention map
#         attn = self.freq_conv(amp)                   # B,1,H,W
#         attn = torch.sigmoid(attn)                   # [0,1]
#
#         # project to C channels and apply residual-style modulation
#         attn_c = self.attn_project(attn)             # B,C,H,W
#         out = x * (1.0 + attn_c)
#
#         return out


# mini_freq_attn_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniFreqAttn(nn.Module):
    """
    改进版轻量频域注意力（V2）
    - 解决原版退化问题：
        1. 初始化非零，避免训练初期冻结
        2. FFT 保留多通道信息
        3. 输出调制范围可控
        4. 频域特征归一化
    - 使用建议：放在 ResNet stage4 输出特征
    """
    def __init__(self, in_channels, freq_kernel=3):
        super().__init__()
        padding = (freq_kernel - 1) // 2

        # 频域卷积（depthwise / group conv）
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=freq_kernel,
                      padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 投影层
        self.project = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        # 初始化
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.project.bias, 0.0)
        for m in self.freq_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, C, H, W] 经过频域注意力调制
        """
        B, C, H, W = x.shape

        # 1. FFT -> 振幅 (保留多通道信息)
        ffted = torch.fft.fft2(x, norm='ortho')
        amp = torch.abs(ffted)
        amp = torch.log1p(amp)

        # 2. 频域归一化（每通道独立）
        amp = (amp - amp.mean(dim=[2,3], keepdim=True)) / (amp.std(dim=[2,3], keepdim=True) + 1e-5)

        # 3. 频域卷积提取注意力
        attn = self.freq_conv(amp)

        # 4. 1x1 投影
        attn = self.project(attn)

        # 5. 限幅控制调制范围 [-0.3, 0.3]
        attn = torch.tanh(attn) * 0.3

        # 6. 残差调制
        out = x * (1.0 + attn)
        return out.contiguous()    #防止后续拼接或多卡并行时出现 tensor stride 异常
        return out

# 测试脚本
if __name__ == "__main__":
    model = MiniFreqAttn(in_channels=2048)
    x = torch.randn(1, 2048, 7, 7)
    y = model(x)
    print(f"[DEBUG] Output shape: {y.shape}")
