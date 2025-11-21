from torch import nn
from timm.models.layers import DropPath
from torch.nn import functional as F


class MHSA(nn.Module):
    def __init__(self, dim, num_heads, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Sequential(nn.Linear(dim, dim))
        self.k = nn.Sequential(nn.Linear(dim, dim))
        self.v = nn.Sequential(nn.Linear(dim, dim))
        self.proj = nn.Linear(dim, dim)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape # 输入的维度
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C) # B, N, C
        x = self.proj(x)
        return x


class CrossScaleAttention(nn.Module):
    def __init__(self, dim, last_dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))
        self.k = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))
        self.v = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, 1, bias=True))
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)

    def forward(self, x, y):
        B, C, H1, W1 = x.shape  # B, C1, H1, W1
        _, _, H2, W2 = y.shape  # B, C2, H2, W2
        N1 = H1 * W1
        N2 = H2 * W2

        x = self.norm_x(x.reshape(B, C, -1).permute(0, 2, 1)) #
        x = x.permute(0, 2, 1).reshape(B, C, H1, W1)
        y = self.norm_y(y.reshape(B, C, -1).permute(0, 2, 1))
        y = y.permute(0, 2, 1).reshape(B, C, H2, W2)

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)
        q = q.reshape(B, C, -1).permute(0, 2, 1)
        q = q.reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        k = k.reshape(B, C, -1).permute(0, 2, 1)
        k = k.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v = v.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # 注意力机制
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = out.transpose(1, 2).contiguous().reshape(B, N1, C).permute(0, 2, 1).reshape(B, C, H1, W1)
        out = self.proj(out)
        out = out.reshape(B, N1, C)
        return out


class CrossResAttn(nn.Module):
    '''Cross-scale Residual Attention module (CRA)'''
    def __init__(self, dim, last_dim, num_heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.channel_conv = nn.Conv2d(last_dim, dim, 1, 1, 0)
        self.norm1_3 = nn.LayerNorm(dim)
        self.norm2_1 = nn.LayerNorm(dim)
        self.norm2_2 = nn.LayerNorm(dim)
        self.attn1 = CrossScaleAttention(dim, last_dim, num_heads)
        self.attn2 = MHSA(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.feb1 = FEB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.)
        self.feb2 = FEB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=0.)

    def forward(self, x, y):
        B, C, H, W = x.shape
        y = self.channel_conv(y)
        y_up = F.interpolate(y, (H, W), mode='bilinear').reshape(B, C, -1).permute(0, 2, 1)
        fuse = y_up + self.drop_path(self.attn1(x, y))
        fuse = fuse + self.drop_path(self.feb1(self.norm1_3(fuse), H, W))
        fuse = fuse + self.drop_path(self.attn2(self.norm2_1(fuse)))
        out = fuse + self.drop_path(self.feb2(self.norm2_2(fuse), H, W))
        return out


class FEB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x
