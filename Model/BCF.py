# import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
# import L1_fusion


import torch
import numpy as np
device = torch.device("cuda"if torch.cuda.is_available()else"cpu")
EPSILON = 1e-5

def row_vector_fusion(tensor1, tensor2, p_type):
    shape = tensor1.size()
    # calculate row vector attention
    row_vector_p1 = row_vector_attention(tensor1, p_type)
    row_vector_p2 = row_vector_attention(tensor2, p_type)

    # get weight map
    row_vector_p_w1 = torch.exp(row_vector_p1) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)
    row_vector_p_w2 = torch.exp(row_vector_p2) / (torch.exp(row_vector_p1) + torch.exp(row_vector_p2) + EPSILON)

    row_vector_p_w1 = row_vector_p_w1.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w1 = row_vector_p_w1.to(device)
    row_vector_p_w2 = row_vector_p_w2.repeat(1, 1, shape[2], shape[3])
    row_vector_p_w2 = row_vector_p_w2.to(device)

    tensor_f = row_vector_p_w1 * tensor1 + row_vector_p_w2 * tensor2

    return tensor_f


def column_vector_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    column_vector_1 = column_vector_attention(tensor1, spatial_type)
    column_vector_2 = column_vector_attention(tensor2, spatial_type)

    column_vector_w1 = torch.exp(column_vector_1) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)
    column_vector_w2 = torch.exp(column_vector_2) / (torch.exp(column_vector_1) + torch.exp(column_vector_2) + EPSILON)

    column_vector_w1 = column_vector_w1.repeat(1, shape[1], 1, 1)
    column_vector_w1 = column_vector_w1.to(device)
    column_vector_w2 = column_vector_w2.repeat(1, shape[1], 1, 1)
    column_vector_w2 = column_vector_w2.to(device)

    tensor_f = column_vector_w1 * tensor1 + column_vector_w2 * tensor2

    return tensor_f

def row_vector_attention(tensor, type="l1_mean"):
    shape = tensor.size()

    c = shape[1]
    h = shape[2]
    w = shape[3]
    row_vector = torch.zeros(1, c, 1, 1)
    if type is"l1_mean":
        row_vector = torch.norm(tensor, p=1, dim=[2, 3], keepdim=True) / (h * w)
    elif type is"l2_mean":
        row_vector = torch.norm(tensor, p=2, dim=[2, 3], keepdim=True) / (h * w)
    elif type is "linf":
            for i in range(c):
                tensor_1 = tensor[0,i,:,:]
                row_vector[0,i,0,0] = torch.max(tensor_1)
            ndarray = tensor.cpu().numpy()
            max = np.amax(ndarray,axis=(2,3))
            tensor = torch.from_numpy(max)
            row_vector = tensor.reshape(1,c,1,1)
            row_vector = row_vector.to(device)
    return row_vector


def column_vector_attention(tensor, type='l1_mean'):

    shape = tensor.size()
    c = shape[1]
    h = shape[2]
    w = shape[3]
    column_vector = torch.zeros(1, 1, 1, 1)
    if type is 'l1_mean':
        column_vector = torch.norm(tensor, p=1, dim=[1], keepdim=True) / c
    elif type is"l2_mean":
        column_vector = torch.norm(tensor, p=2, dim=[1], keepdim=True) / c
    elif type is "linf":
        column_vector, indices = tensor.max(dim=1, keepdim=True)
        column_vector = column_vector / c
        column_vector = column_vector.to(device)
    return column_vector

def attention_fusion_weight(tensor1, tensor2, p_type='l1_mean'):
    f_row_vector = row_vector_fusion(tensor1, tensor2, p_type)
    f_column_vector = column_vector_fusion(tensor1, tensor2, p_type)

    tensor_f = (f_row_vector + f_column_vector)

    return tensor_f

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Modulation(nn.Module):
    def __init__(self):
        super(Modulation, self).__init__()
        self.scale_conv0 = nn.Conv2d(64, 64, kernel_size=1)
        self.scale_conv1 = nn.Conv2d(64, 64, kernel_size=1)
        self.shift_conv0 = nn.Conv2d(64, 64, kernel_size=1)
        self.shift_conv1 = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x, y):
        scale = self.scale_conv1(F.leaky_relu(self.scale_conv0(y), 0.1, inplace=True))
        shift = self.shift_conv1(F.leaky_relu(self.shift_conv0(y), 0.1, inplace=True))
        return x * (scale + 1) + shift



class SCAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5

        self.norm_l = LayerNorm(c, 'my_layernorm')
        self.norm_r = LayerNorm(c, 'my_layernorm')

        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r


class FusionNet(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(FusionNet, self).__init__()
        self.scam = SCAM(dim)
        self.l1_fusion = attention_fusion_weight


    def forward_features(self, x, y):
        x_catt, y_catt = self.scam(x, y)
        x = x + x_catt
        y = y + y_catt
        return x, y

    def forward(self, x, y):
        x, y = self.forward_features(x, y)
        out = self.l1_fusion(x, y)
        return out

if __name__ == '__main__':
    from thop import profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionNet(384, 8, 2, False, 'my_layernorm').to(device)
    x = torch.randn(1, 384, 64, 64).to(device)
    y = torch.randn(1, 384, 64, 64).to(device)
    out = model(x, y)
    print(out.shape)
    flops, params = profile(model, inputs=(x,y))
    print('   Number of parameters: %.5fM' % (params / 1e6))  #  0.59136M
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))   #  2.41592G