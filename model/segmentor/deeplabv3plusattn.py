import model.backbone.resnet as resnet
from model.segmentor.layers import DynamicMultiHeadConv
from model.util.feature_noise import exponential_noise, dropout_noise, salt_pepper_noise, gaussian_noise, uniform_noise
from util.uncertainty import compute_uncertainty

import torch
from torch import nn
import torch.nn.functional as F
import random

import pdb

class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                         replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.drop_ratio = cfg['drop_ratio']

    def forward(self, x1, x2, feat_noise=False):
        h, w = x1.shape[-2:]

        feats1 = self.backbone.base_forward(x1)
        c11, c14 = feats1[0], feats1[-1]
        
        feats2 = self.backbone.base_forward(x2)
        c21, c24 = feats2[0], feats2[-1]

        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs()
        if feat_noise:
            # select noise in: dropout, salt_pepper, gaussian
            outs = self._decode(torch.cat((c1, dropout_noise(c1))),
                                torch.cat((c4, dropout_noise(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp
        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


class DeepLabV3PlusAttn(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3PlusAttn, self).__init__()

        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                         replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        self.chann_attn_conv = DynamicMultiHeadConv(high_channels // 8 + 48, high_channels // 8 + 48, kernel_size=3, stride=1, padding=1, heads=4, squeeze_rate=16, gate_factor=0.25)
        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)
        self.drop_ratio = cfg['drop_ratio']

    def forward(self, x1, x2, feat_noise=False, forward_analyze=False, uncertainty=False):

        h, w = x1.shape[-2:]

        feats1 = self.backbone.base_forward(x1)
        c11, c14 = feats1[0], feats1[-1]
        
        feats2 = self.backbone.base_forward(x2)
        c21, c24 = feats2[0], feats2[-1]

        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs()

        if feat_noise:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(self.drop_ratio)(c1))),
                                torch.cat((c4, nn.Dropout2d(self.drop_ratio)(c4))), True)
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)
            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def _decode(self, c1, c4, feat_noise=False):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        if feat_noise:
            feature = self.fuse(self.chann_attn_conv(feature)[0])
            out = self.classifier(feature)
            return out
        
        feature = self.fuse(feature)
        out = self.classifier(feature)
        return out
    
    def forward_base(self, x1, x2):
        feats1 = self.backbone.base_forward(x1)
        c11, c14 = feats1[0], feats1[-1]
        
        feats2 = self.backbone.base_forward(x2)
        c21, c24 = feats2[0], feats2[-1]

        c1 = (c11 - c21).abs()
        c4 = (c14 - c24).abs()
        return c1, c4
    
    def decoder(self, c1, c4, h=256, w=256):
        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out

