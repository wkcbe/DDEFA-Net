import torch
import torch.nn as nn
from bakebone.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
from .MSFA import MSFA
from .BCF import FusionNet
from .DDCR import CAViT

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # focal
        self.focal_encoder = pvt_v2_b2()
        # Fusion
        self.Fusion1 = FusionNet(384, 8, 2, False, 'my_layernorm')
        self.Fusion2 = FusionNet(384, 8, 2, False, 'my_layernorm')
        self.Fusion3 = FusionNet(384, 8, 2, False, 'my_layernorm')
        self.Fusion4 = FusionNet(384, 8, 2, False, 'my_layernorm')

        # MSFA
        self.msfa1 = MSFA(384)
        self.msfa2 = MSFA(384)
        self.msfa3 = MSFA(384)
        # Enhance
        self.CAViT4 = CAViT(512, 128, img_size=8, patch_size=1, embed_dim=128, groups=1)
        self.CAViT3 = CAViT(320, 128, img_size=16, patch_size=1, embed_dim=128, groups=1)
        self.CAViT2 = CAViT(128, 64, img_size=32, patch_size=1, embed_dim=64, groups=1)
        self.CAViT1 = CAViT(64, 32, img_size=64, patch_size=1, embed_dim=32, groups=1)

        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.bn2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True)

        conv = nn.Sequential()
        conv.add_module('conv1', nn.Conv2d(384, 96, 3, 1, 1))
        conv.add_module('bn1', self.bn1)
        conv.add_module('relu1', nn.ReLU(inplace=True))
        conv.add_module('conv2', nn.Conv2d(96, 1, 3, 1, 1))
        conv.add_module('bn2', self.bn2)
        conv.add_module('relu2', nn.ReLU(inplace=True))
        self.conv = conv

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(384, 384, kernel_size=3, padding=1))
        self.conv_last = nn.Conv2d(384, 1, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(32, 384, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(32, 384, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 384, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(32, 384, kernel_size=1, padding=0)

        # rgb
        self.rgb_encoder = pvt_v2_b2()
        # 输出通道数为32
        self.rfb33 = CAViT(512, 128, img_size=8, patch_size=1, embed_dim=128, groups=1)
        self.rfb22 = CAViT(320, 128, img_size=16, patch_size=1, embed_dim=128, groups=1)
        self.rfb11 = CAViT(128, 64, img_size=32, patch_size=1, embed_dim=64, groups=1)
        self.rfb00 = CAViT(64, 32, img_size=64, patch_size=1, embed_dim=32, groups=1)

    def forward(self, x, y):

        # focal
        ba = x.size()[0]//12
        x = self.focal_encoder(x)  # x:{12,(64,128,320,512),(64,32,16,8),(64,32,16,8)}
        # 进一步提取特征，并统一调整通道数
        x0q = self.CAViT1(x[0])
        x1q = self.CAViT2(x[1])
        x2q = self.CAViT3(x[2])
        x3q = self.CAViT4(x[3])
        # rgb
        y = self.rgb_encoder(y)

        y[0] = self.rfb00(y[0])  # [1, 32, 64, 64]
        y[1] = self.rfb11(y[1])  # [1, 32, 32, 32]
        y[2] = self.rfb22(y[2])  # [1, 32, 16, 16]
        y[3] = self.rfb33(y[3])  # [1, 32, 8, 8]

        out_xq = x0q

        x0q_sal = torch.cat(torch.chunk(x0q.unsqueeze(1), ba, dim=0), dim=1)    # [12, 1, 32, 64, 64]
        x0a = torch.cat(torch.chunk(x0q_sal, 12, dim=0), dim=2).squeeze(0)  # [1, 384, 64, 64]
        # x0q_sal = self.conv(x0a)    # stage1特征图的映射为显著图与GT计算损失 [1, 1, 64, 64]

        x1q_sal = torch.cat(torch.chunk(x1q.unsqueeze(1), ba, dim=0), dim=1)    # [12, 1, 32, 32, 32]
        x1a = torch.cat(torch.chunk(x1q_sal, 12, dim=0), dim=2).squeeze(0)  # [1, 384, 32, 32]
        # x1q_sal = self.conv(x1a)                                            # [1, 1, 32, 32]

        x2q_sal = torch.cat(torch.chunk(x2q.unsqueeze(1), ba, dim=0), dim=1)    # [12, 1, 32, 16, 16]
        x2a = torch.cat(torch.chunk(x2q_sal, 12, dim=0), dim=2).squeeze(0)  # [1, 384, 16, 16]
        # x2q_sal = self.conv(x2a)                                            # [1, 1, 16, 16]

        x3q_sal = torch.cat(torch.chunk(x3q.unsqueeze(1), ba, dim=0), dim=1)    # [12, 1, 32, 8, 8]
        x3a = torch.cat(torch.chunk(x3q_sal, 12, dim=0), dim=2).squeeze(0)  # [1, 384, 8, 8],384=32*12
        # x3q_sal = self.conv(x3a)  # [1, 1, 8, 8]

        y[2] = self.conv2(y[2])
        y[3] = self.conv1(y[3])  # [1, 384, 8, 8]
        y[1] = self.conv3(y[1])
        y[0] = self.conv4(y[0])
        out_xf = y[0]

        # 跨模态特征融合
        xy3 = self.Fusion2(x3a, y[3])  # [1, 384, 8, 8],[1, 32, 8, 8]
        x3q_sal = self.conv(xy3)  # [1, 1, 8, 8]
        xy2 = self.Fusion1(x2a, y[2])  #  [1, 384, 16, 16],[1, 32, 16, 16]
        xy1 = self.Fusion3(x1a, y[1])
        xy0 = self.Fusion4(x0a, y[0])  #  [1, 384, 64, 64],[1, 32, 64, 64]

        # 由低到高上采样,采用多尺度融合策略合并多尺度特征图
        xy_3 = F.interpolate(xy3, scale_factor=2, mode='bilinear', align_corners=False)  # [1, 384, 16, 16]
        xy23 = self.msfa1(xy_3, xy2)  # [1, 384, 16, 16]
        x2q_sal = self.conv(xy23)

        xy23 = F.interpolate(xy23, scale_factor=2, mode='bilinear',align_corners=False)  # [1,384,16,16]--->[1,384,32,32]
        xy123 = self.msfa2(xy23, xy1)  # [1, 384, 32, 32]
        x1q_sal = self.conv(xy123)

        xy123 = F.interpolate(xy123, scale_factor=2, mode='bilinear',align_corners=False)  # [1,384,32,32]--->[1,384,64,64]
        xy0123 = self.msfa3(xy123, xy0)  # [1, 384, 64, 64],融合了其他stage多尺度信息并完成多模态融合操作后的特征图
        x0q_sal = self.conv(xy0123)    # stage1特征图的映射为显著图与GT计算损失 [1, 1, 64, 64]

        # 通过卷积层合并不同切片减少通道数,并通过采样到GT获得最终的预测显著图
        fuse_sal = self.conv_last(xy0123)  # [1, 1, 64, 64]
        fuse_pred = F.interpolate(fuse_sal, size=(256, 256), mode='bilinear', align_corners=False)  # [2, 1, 256, 256]

        return x0q_sal, x1q_sal, x2q_sal, x3q_sal, fuse_pred, out_xf, out_xq, xy0123, fuse_sal


if __name__ == '__main__':
    from thop import profile
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model().to(device)
    x = torch.randn(12, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    out = model(x,y)
    # print(out.shape)
    flops, params = profile(model, inputs=(x,y))
    print('   Number of parameters: %.5fM' % (params / 1e6))  # 0.01389M
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))  # 0.05729G