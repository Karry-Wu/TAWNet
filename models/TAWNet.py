
import torch
from torch import nn
from torch.nn import functional as F
from thop import profile
from models.encoder import CBR, P2tBackbone
from models.CRA import CrossResAttn


class TAWNet(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        layer_scale_init_value = 1e-6
        self.backbone = P2tBackbone(path)

        self.dqo4 = DQO(512, 512)
        self.dqo3 = DQO(320, 320)
        self.dqo2 = DQO(128, 128)
        self.dqo1 = DQO(64, 64)

        self.rfp4 = RFPU(num_channels=512, int_channels=1344)
        self.rfp3 = RFPU(num_channels=320, int_channels=960)
        self.rfp2 = RFPU(num_channels=128, int_channels=512)
        self.rfp1 = RFPU(num_channels=64, int_channels=256)
        self.dfp4 = DFPU(num_channels=512, int_channels=1344)
        self.dfp3 = DFPU(num_channels=320, int_channels=960)
        self.dfp2 = DFPU(num_channels=128, int_channels=512)
        self.dfp1 = DFPU(num_channels=64, int_channels=256)

        self.tawf4 = TAWF(512)
        self.tawf3 = TAWF(320)
        self.tawf2 = TAWF(128)
        self.tawf1 = TAWF(64)

        self.conv3_2 = CBR(320, 320, 3, 1, 1, act=nn.PReLU())
        self.cra3 = CrossResAttn(320, 512, 5)

        self.conv2_2 = CBR(128, 128, 3, 1, 1, act=nn.PReLU())
        self.cra2 = CrossResAttn(128, 320, 2)

        self.conv1_2 = CBR(64, 64, 3, 1, 1, act=nn.PReLU())
        self.cra1 = CrossResAttn(64, 128, 1)

        self.side_out4 = CBR(512, 1, 3, 1, 1, act=nn.PReLU())
        self.side_out3 = CBR(320, 1, 3, 1, 1, act=nn.PReLU())
        self.side_out2 = CBR(128, 1, 3, 1, 1, act=nn.PReLU())
        self.ps_out = nn.PixelShuffle(4)
        self.out_conv = CBR(64, 16, 3, 1, 1, act=nn.PReLU())
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((64)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, y):
        x_out, y_out = self.backbone(x, y)
        x1, x2, x3, x4 = x_out  # RGB
        y1, y2, y3, y4 = y_out  # depth

        y1 = self.dqo1(y1)
        y2 = self.dqo2(y2)
        y3 = self.dqo3(y3)
        y4 = self.dqo4(y4)

        x1 = self.rfp1(x1, x2)
        x2 = self.rfp2(x3, x2, x1)
        x3 = self.rfp3(x4, x3, x2)
        x4 = self.rfp4(x4, x3)
        y1 = self.dfp1(y1, y2)
        y2 = self.dfp2(y3, y2, y1)
        y3 = self.dfp3(y4, y3, y2)
        y4 = self.dfp4(y4, y3)

        # layer 4
        fuse4 = self.tawf4(x4, y4)
        side_out4 = self.side_out4(fuse4)
        # layer3
        B, C, H, W = x3.shape
        fuse3 = self.tawf3(x3, y3)
        cs_fuse3 = self.cra3(fuse3, fuse4)
        cs_fuse3 = cs_fuse3.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse3 = self.conv3_2(fuse3 * cs_fuse3)
        side_out3 = self.side_out3(fuse3)
        # layer 2
        B, C, H, W = x2.shape
        fuse2 = self.tawf2(x2, y2)
        cs_fuse2 = self.cra2(fuse2, fuse3)
        cs_fuse2 = cs_fuse2.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse2 = self.conv2_2(fuse2 * cs_fuse2)
        side_out2 = self.side_out2(fuse2)
        # layer1
        B, C, H, W = x1.shape
        fuse1 = self.tawf1(x1, y1)
        cs_fuse1 = self.cra1(fuse1, fuse2)
        cs_fuse1 = cs_fuse1.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        fuse1 = self.conv1_2(fuse1 * cs_fuse1)
        fuse1 = fuse1.reshape(B, C, -1).permute(0, 2, 1)
        fuse1 = self.gamma * fuse1
        fuse1 = fuse1.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # fuse1 = self.er1(fuse1)
        side_out1 = self.out_conv(fuse1)
        side_out1 = self.ps_out(side_out1)
        return side_out4, side_out3, side_out2, side_out1


class DQO(nn.Module):
    """Depth Quality Optimization module (DQO)"""
    def __init__(self, inch, ouch):
        super().__init__()
        self.Rconv = nn.Sequential(nn.Conv2d(inch, ouch, 3, 1, 1), nn.PReLU())
        self.sigmoid = nn.Sigmoid()

    def forward(self, fdepth):
        f1 = self.Rconv(fdepth)
        max_out, _ = torch.max(f1 * fdepth, dim=1, keepdim=True)
        wc = self.sigmoid(max_out)
        out = wc * f1 + f1
        return out


class RFPU(nn.Module):
    """RGB feature purification unit"""
    def __init__(self, num_channels,int_channels):
        super(RFPU, self).__init__()
        self.conv = nn.Conv2d(int_channels, num_channels,kernel_size=1,stride=1)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(int_channels)

    def forward(self, in1, in2=None, in3=None):
        if in2 !=None and in3 !=None:
            in1 = F.interpolate(in1, size=in2.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in2.size()[2:], mode='bilinear')
        elif in2!=None and in3==None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = in1
        x = torch.cat((in1, in2, in3), 1)
        ca = self.ca(x)
        ca = self.conv(ca)
        out = self.sa(ca)
        out = self.sa_conv(out) * ca
        return out


class DFPU(nn.Module):
    """Depth feature purification unit"""
    def __init__(self, num_channels,int_channels):
        super(DFPU, self).__init__()
        self.conv = nn.Conv2d(int_channels, num_channels,kernel_size=1,stride=1)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(int_channels)

    def forward(self, in1, in2=None, in3=None):
        if in2 !=None and in3 !=None:
            in1 = F.interpolate(in1, size=in2.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in2.size()[2:], mode='bilinear')
        elif in2!=None and in3==None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = in1
        x = torch.cat((in1, in2, in3), 1)
        sa = self.sa(x)
        sa = self.sa_conv(sa)
        out = self.ca(x.mul(sa))
        out = self.conv(out) * sa
        return out


class TAWF(nn.Module):
    def __init__(self, channels):
        """
        Three-dimensional Adaptive Weighted Fusion module (TAWF)
        """
        super().__init__()
        self.rgb_input = nn.Conv2d(channels, 1,1)
        self.depth_input = nn.Conv2d(channels, 1, 1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.ic_conv = nn.Conv2d(2 * channels, 2, kernel_size=1, padding=0)
        self.final_fc = nn.Linear(channels * 3, 3)

    def compute_cosine_similarity(self, rgb_feat, depth_feat):
        bs, c, h, w = rgb_feat.shape
        rgb_feat_cosine = self.rgb_input(rgb_feat).view(bs, -1)
        depth_feat_cosine = self.depth_input(depth_feat).view(bs, -1)
        cosine_sim = self.cosine_similarity(rgb_feat_cosine, depth_feat_cosine)
        cosine_sim = (1 - cosine_sim) / 2.0
        cosine_sim = 1 - cosine_sim
        return cosine_sim

    def forward(self, fr, fd):
        # semantic similarity
        bs = fr.shape[0]
        cosine_sim = self.compute_cosine_similarity(fr, fd)
        F_ss = cosine_sim.view(bs, 1, 1, 1) * fd
        # modal difference
        F_md = fr - fd
        # information complementarity
        cat_rd = torch.cat([fr, fd], dim=1)
        weights = self.ic_conv(cat_rd)
        weights_rd = F.softmax(weights, dim=1)
        w_r, w_d = weights_rd.chunk(2, dim=1)
        F_ic = (fr * w_r) + (fd * w_d)
        F_cat = torch.cat([F_md, F_ss, F_ic], dim=1)

        gmp = F.adaptive_max_pool2d(F_cat, (1, 1)).view(F_cat.size(0), -1)
        avg = F.adaptive_avg_pool2d(F_cat, (1, 1)).view(F_cat.size(0), -1)
        gp_out = gmp + avg
        final_weights = self.final_fc(gp_out)
        w_md, w_ss, w_ic = F.softmax(final_weights, dim=1).chunk(3, dim=1)

        w_md = w_md.reshape(-1, 1, 1, 1)
        w_ss = w_ss.reshape(-1, 1, 1, 1)
        w_ic = w_ic.reshape(-1, 1, 1, 1)
        F_fuse = (F_md * w_md) + (F_ss * w_ss) + (F_ic * w_ic)
        return F_fuse


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = min(8,in_planes // ratio)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # PAT94
        self.fc1 = nn.Conv2d(in_planes *2, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca = self.fc2(self.relu1(self.fc1(torch.cat([self.max_pool(x),self.ave_pool(x)],dim=1))))  # PAT94--111
        out = x * self.sigmoid(ca)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':

    print('==> Building model..')
    model = TAWNet()
    RGB_input = torch.randn(1, 3, 384, 384)
    DEP_input = torch.randn(1, 3, 384, 384)
    flops, params = profile(model, (RGB_input, DEP_input))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
