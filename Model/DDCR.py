import torch
import torch.nn as nn
from .SpectFormer import SpectFormer

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class LocalContextExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(out_channels//reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, _, _ = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)

class CAViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size=64, patch_size=1, embed_dim=64, groups=1):
        super(CAViT, self).__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_last = nn.Conv2d(embed_dim, 32, 3, 1, 1)
        # 局部路径
        self.local_conv = LocalContextExtractor(self.in_channels, self.out_channels)
        # 全局路径
        self.glabal = SpectFormer(img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)

    def forward(self, x):
        # x:1,32,8,8
        # 局部特征处理
        local_features = self.local_conv(x)  # 12,64,64,64
        # 全局特征处理 - FFT
        global_features = self.glabal(x)  # 12,64,64,64
        # 元素级加和
        output = self.conv_last(global_features + local_features)
        return output

if __name__ == '__main__':
    from thop import profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 注意：embed_dim需要与局部特征处理分支的输出通道数一致
    block = CAViT(64, 32, img_size=64, patch_size=1, embed_dim=32, groups=1).to(device)
    input = torch.rand(1, 64, 64, 64).to(device)
    output = block(input)
    print(output.shape)
    flops, params = profile(block, inputs=(input,))
    print('   Number of parameters: %.5fM' % (params / 1e6))  # 0.02719M
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))  # 0.10958G
