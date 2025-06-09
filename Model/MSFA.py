import torch
from torch import nn
from einops.layers.torch import Rearrange

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True), )

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True), )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn

class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CGAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAttention, self).__init__()
        self.conv1 = conv3x3_bn_relu(dim, dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = default_conv(dim, dim, 3, bias=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # res = self.act1(self.conv1(x)) + x
        # res = self.conv2(res)
        res = x
        pattn1 = self.ca(res) + self.sa(res)
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2 + x
        res = self.sigmoid(res)
        return res


class MSFA(nn.Module):
    def __init__(self, dim, reduction=8):
        super(MSFA, self).__init__()
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = conv3x3_bn_relu(dim, dim)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fuse_high, fuse_low):
        # fuse_high = self.conv1(self.up1(fuse_high))  # fuse_high为上采样后的结果
        initial = fuse_high + fuse_low
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.pa(initial, pattn1)
        result = initial + pattn2 * fuse_high + (1 - pattn2) * fuse_low
        result = self.sigmoid(self.conv2(result))
        return result


if __name__ == '__main__':
    from thop import profile
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MSFA(32).to(device)
    x = torch.randn(1, 32, 32, 32).to(device)
    y = torch.randn(1, 32, 64, 64).to(device)
    out = model(x,y)
    print(out.shape)
    flops, params = profile(model, inputs=(x,y))
    print('   Number of parameters: %.5fM' % (params / 1e6))  # 0.01389M
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))  # 0.05729G
