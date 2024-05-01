from typing import Dict
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.cbam = CBAMLayer(in_channels)

    def forward(self, x):

        x1 = self.cbam(x) + x
        return self.maxpool_conv(x1)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SE_Block(nn.Module):

   def __init__(self, in_planes, k_size=3):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

   def forward(self, x):


        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv_1x1_output = nn.Sequential(
            nn.Conv2d(out_channel * 5, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.senet = SE_Block(in_planes=out_channel * 5)
    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)

        feature_cat = torch.cat([image_features, conv1, conv2, conv3, conv4], dim=1)
        # 加入se注意力机制
        se_aspp = self.senet(feature_cat)
        se_feature_cat = se_aspp * feature_cat
        net = self.conv_1x1_output(se_feature_cat)
        return net

class CFF(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(CFF, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        ## ---------------------------------------- ##

        self.layer1 = ASPP(out_channel//2, out_channel//2)

        self.layer2 = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer3 = BasicConv2d(in_channel2, out_channel // 2, 1)

        self.layer3_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        # self.layer3_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
        #                               nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer5_1 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                                      nn.BatchNorm2d(out_channel // 2), act_fn)
        # self.layer5_2 = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
        #                               nn.BatchNorm2d(out_channel // 2), act_fn)

        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(out_channel), act_fn)

    def forward(self, x0, x1):
        ## ------------------------------------------------------------------ ##

        x0_1 = self.layer2(x0)
        x1_1 = self.layer3(x1)
        # print("X0_1:", x0_1.shape)
        # print("X1_1:", x1_1.shape)
        x_3_1 = self.layer3_1(torch.cat((x0_1, x1_1), dim=1))
        x_5_1 = self.layer5_1(torch.cat((x1_1, x0_1), dim=1))
        # print("X_3_1:", x_3_1.shape)
        # print("X_5_1:", x_5_1.shape)

        x6 = self.layer1(torch.mul(x_3_1, x_5_1))
        # print("X6:",x6.shape)
        out = self.layer_out(x0_1 + x1_1 + x6)

        return out

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()


        self.convpro = ConvPro(in_channels, out_channels)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = DoubleConv(out_channels, out_channels, in_channels//2)


        self.feature_fusion = CFF(out_channels, out_channels,  out_channels)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        x1 = self.up(x1)

        x1 = self.convpro(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # x = torch.cat([x2, x1], dim=1)
        x = self.feature_fusion(x1,x2)
        x = self.conv(x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):

        return self.conv(x)

class ConvPro(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        channels_per_group = out_channels // 4

        self.depthwise_conv1 = nn.Conv2d(in_channels, channels_per_group, kernel_size=3, padding=1, stride=1, groups=channels_per_group)
        self.pointwise_conv1 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=1, stride=1)

        self.depthwise_conv2 = nn.Conv2d(in_channels, channels_per_group, kernel_size=5, padding=2, stride=1, groups=channels_per_group)
        self.pointwise_conv2 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=1, stride=1)

        self.depthwise_conv3 = nn.Conv2d(in_channels, channels_per_group, kernel_size=7, padding=3, stride=1, groups=channels_per_group)
        self.pointwise_conv3 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=1, stride=1)

        self.depthwise_conv4 = nn.Conv2d(in_channels, channels_per_group, kernel_size=9, padding=4, stride=1, groups=channels_per_group)
        self.pointwise_conv4 = nn.Conv2d(channels_per_group, channels_per_group, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels, channels_per_group, kernel_size=1, stride=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_per_group, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_6 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv_5(x)

        x1 = self.depthwise_conv1(x)
        x1 = self.relu(x1)
        x1 = self.pointwise_conv1(x1)

        x2 = self.depthwise_conv2(x)
        x2 = self.relu(x2)
        x2 = self.pointwise_conv2(x2)

        x3 = self.depthwise_conv3(x)
        x3 = self.relu(x3)
        x3 = self.pointwise_conv3(x3)

        x4 = self.depthwise_conv4(x)
        x4 = self.relu(x4)
        x4 = self.pointwise_conv4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_6(self.relu(x))

        out = self.final_conv(x + y)
        return out




class MCGFF(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 base_c: int = 64):
        super(MCGFF, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = DoubleConv(in_channels, base_c)

        self.down1 = Down(base_c, base_c * 2)

        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)

        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16, base_c * 8)
        self.up2 = Up(base_c * 8, base_c * 4)
        self.up3 = Up(base_c * 4, base_c * 2)
        self.up4 = Up(base_c * 2, base_c)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits

if __name__ == '__main__':
    image = torch.randn(3, 3, 352, 352).cuda()
    model = MCGFF(in_channels=3, num_classes=1).cuda()
    output = model(image)
    print(output.shape)