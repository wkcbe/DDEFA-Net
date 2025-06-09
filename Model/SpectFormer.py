import math
import logging
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft


_logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)  # 12,256,128
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 12,256,32
        x = self.drop(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
    def forward(self, x, spatial_size=None):
        B, N, C = x.shape  # N=H*W,(12,4096,64)
        if spatial_size is None:
            a = b = int(math.sqrt(N))  # a=b=64,64*64=4096
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)  # (12,64,64,64)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')  # (12,64,33,64)
        weight = torch.view_as_complex(self.complex_weight)  # (64,33,64)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')   # (12,64,64,64)

        x = x.reshape(B, N, C)  # (12,4096,64)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = self.filter(self.norm1(x))  # 12,256,32
        x = self.mlp(self.norm2(x))  # 12,256,32
        x = x + self.drop_path(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x,H,W

# x(N,C,H,W):{12,(64,128,320,512),(64,32,16,8),(64,32,16,8)}
class SpectFormer(nn.Module):

    def __init__(self, img_size=64, patch_size=8, in_chans=64, embed_dim=4096, depth=1,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None,
                 dropcls=0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        alpha = 4
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                          norm_layer=norm_layer, h=h, w=w)
            self.blocks.append(layer)


        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()



        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)  # 12,64*64,64
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # 12,4096,64
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # 12,64,64,64
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        return x
if __name__ == '__main__':
    # 注意：此时的设计模式
    # 输入输出HW不发生改变，输出特征图通道数C有两种情况
    # C由embed_dim绝定：(patch_size=1,embed_dim=N)--->C=embed_dim
    # C由embed_dim和patch_size共同决定：(patch_size>=2,embed_dim=N)--->C=embed_dim/patch_size^2
    block = SpectFormer(img_size=64, patch_size=1, in_chans=64, embed_dim=64, depth=1)
    input = torch.rand(12, 64, 64, 64)
    output = block(input)
    print(output.shape)

