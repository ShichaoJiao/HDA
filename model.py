import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
                                  nn.InstanceNorm2d(in_channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels, in_channels, 3, padding=1, padding_mode='reflect'),
                                  nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.conv(x)


class Generator(nn.Module):
    def __init__(self, in_channels=64, num_block=9):
        super(Generator, self).__init__()

        self.n_view = 12
        # in conv
        self.in_conv = nn.Sequential(nn.Conv2d(3, in_channels, 7, padding=3, padding_mode='reflect'),
                                     nn.InstanceNorm2d(in_channels), nn.ReLU(inplace=True))

        # down sample
        down_sample = []
        for _ in range(2):
            out_channels = in_channels * 2
            down_sample += [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                            nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.01, inplace=True)]
            in_channels = out_channels
        self.down_sample = nn.Sequential(*down_sample)

        # conv blocks
        self.convs = nn.Sequential(*[ResidualBlock(in_channels) for _ in range(num_block)])

        # up sample
        up_sample = []
        for _ in range(2):
            out_channels = in_channels // 2
            up_sample += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                          nn.InstanceNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            in_channels = out_channels
        self.up_sample = nn.Sequential(*up_sample)

        # out conv
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels, 3, 7, padding=3, padding_mode='reflect'), nn.Tanh())

    def forward(self, x,is_target = 1):
        if is_target:
            view_batch = x
            view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
            if torch.isnan(view_batch).any() or torch.isinf(view_batch).any():
                print("Input contains NaN or Inf!")
    
            x = self.in_conv(view_batch)
            x = self.down_sample(x)
            x = self.convs(x)
            x = self.up_sample(x)
            out = self.out_conv(x)
            out = out.view(-1, self.n_view, view_batch.size(1), view_batch.size(2), view_batch.size(3))
        else:
            x = self.in_conv(x)
            x = self.down_sample(x)
            x = self.convs(x)
            x = self.up_sample(x)
            out = self.out_conv(x)
        # if torch.isnan(out).any():
        #     print()
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=64):
        super(Discriminator, self).__init__()
        self.n_view = 12

        self.conv1 = nn.Sequential(nn.Conv2d(3, in_channels, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels * 2, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 2), nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels * 4, 4, stride=2, padding=1),
                                   nn.InstanceNorm2d(in_channels * 4), nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels * 4, in_channels * 8, 4, padding=1),
                                   nn.InstanceNorm2d(in_channels * 8), nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Conv2d(in_channels * 8, 1, 4, padding=1)

    def forward(self, x,is_target = 1):
        if is_target:
            view_batch = x
            view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
            x = self.conv1(view_batch)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            out = self.conv5(x)
            out = out.view(-1, self.n_view, out.size(1), out.size(2), out.size(3))
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            out = self.conv5(x)
        return out


class Extractor(nn.Module):
    def __init__(self, backbone_type, emb_dim):
        super(Extractor, self).__init__()

        # backbone
        model_name = 'resnet50' if backbone_type == 'resnet50' else 'vgg16'
        pretrained_cfg = timm.create_model(model_name).default_cfg
        # pretrained_cfg['file'] = '/root/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth'
        pretrained_cfg['file'] = r'C:\Users\jiao\.cache\torch\hub\checkpoints\resnet50_a1_0-14fe96d1.pth'
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=emb_dim, global_pool='max', pretrained_cfg=pretrained_cfg)
        self.n_view = 12

    def forward(self, x,is_target = 1):
        if is_target:
            view_batch = x
            view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
            view_fts = self.backbone(view_batch)
            view_fts = view_fts.view(view_fts.size(0), -1)
            local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
            x, _ = local_view_fts.max(dim=1)
        else:
            x = self.backbone(x)
        out = F.normalize(x, dim=-1)
        return out


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()