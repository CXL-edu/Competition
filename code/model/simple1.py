import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 39*2=78 -> 32
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
        self.conv5 = nn.Conv2d(32, 32, kernel_size=1, bias=False, padding_mode='replicate')
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.conv2(x)
        x = self.relu3(self.bn3(self.conv3(self.upsample(x1))))
        x = self.avgpool(self.relu4(self.bn4(self.conv4(x))))
        x = x + x1
        x = self.relu5(self.bn5(self.conv5(x)))

        return x


class SubModel(nn.Module):
    def __init__(self, in_channels, out_channels=20):
        super().__init__()
        # 32 -> 20
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True, padding_mode='replicate')

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class SubModel4(nn.Module):
    def __init__(self, in_channels, out_channels=20):
        super().__init__()
        # 32 -> 20
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True, padding_mode='replicate')

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x



# class Model(nn.Module):
#     def __init__(self, in_channel=39*2):
#         super().__init__()
#         self.in_channel = in_channel
#         self.encoder = Encoder(self.in_channel, 32)
#         self.sub_model4 = SubModel4(32,20)
#
#         # Encoder 39*2=78 -> 32
#         self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  #[B, 78, 161, 161] -> [B, 64, 161, 161
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    # [B, 32, 161, 161] -> [B, 32, 322, 322]
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 32, 322, 322] -> [B, 32, 322, 322]
#         self.bn3 = nn.BatchNorm2d(32)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False, padding_mode='replicate')  # [B, 32, 322, 322] -> [B, 32, 322, 322]
#         self.bn4 = nn.BatchNorm2d(32)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)    # [B, 32, 322, 322] -> [B, 32, 161, 161]
#         self.conv5 = nn.Conv2d(32, 32, kernel_size=1, bias=False, padding_mode='replicate') # [B, 32, 161, 161] -> [B, 32, 161, 161]
#         self.bn5 = nn.BatchNorm2d(32)
#         self.relu5 = nn.ReLU(inplace=True)
#
#         # SubModel 32 -> 20
#         self.conv1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, 20, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(20)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(20, 20, kernel_size=1, bias=True, padding_mode='replicate')
#
#
#
#     def forward(self, x):
#         x = x.view(-1, self.in_channel, 161, 161)   # b,t,c,h,w -> b,t*c,h,w
#         x = self.encoder(x)    # b,32,161,161
#         output = self.sub_model4(x).unsqueeze(2)    # b,32,161,161 -> b,20,1,161,161
#         return output


import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channel=70 * 2):
        super().__init__()
        self.in_channel = in_channel
        # 70*2=140 -> 20
        self.conv0 = nn.Conv2d(in_channel, 128, kernel_size=7, padding=3, bias=False, padding_mode='replicate')  # [B, 78, 161, 161] -> [B, 64, 161, 161
        self.bn0 = nn.BatchNorm2d(128)
        self.conv0_1 = nn.Conv2d(128, 128, kernel_size=5, padding=2, bias=True, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 64, 161, 161]
        self.conv0_2 = nn.Conv2d(128, 128, kernel_size=7, padding=3, bias=True, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 64, 161, 161]
        self.conv1 = nn.Conv2d(128, 64, kernel_size=7, padding=3, bias=False,
                               padding_mode='replicate')  # [B, 78, 161, 161] -> [B, 64, 161, 161
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True,
                               padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 64, 161, 161]
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False,
                               padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True,
                               padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=True,
                               padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv4 = nn.Conv2d(32, 20, kernel_size=7, padding=3, bias=False,
                               padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 20, 161, 161]
        self.bn3 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=True,
                               padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]
        self.conv6 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=True,
                               padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_channel, 161, 161)  # b,t,c,h,w -> b,t*c,h,w
        x0 = self.bn0(self.conv0(x))
        x = self.relu(x0)
        x = self.relu(self.conv0_1(x))
        x = self.relu(self.conv0_2(x))
        x = x + x0
        x1 = self.bn1(self.conv1(x))
        x = self.relu(x1)  # b,64,161,161\
        x = self.relu(self.conv2_1(x))  # b,64,161,161
        x = x + x1
        x2 = self.bn2(self.conv2(x))
        x = self.relu(x2)  # b,32,161,161
        x = self.relu(self.conv3_1(x))  # b,32,161,161
        x = self.relu(self.conv3(x))  # b,32,161,161
        x = x + x2
        x = self.relu(self.bn3(self.conv4(x)))  # b,20,161,161
        # x = torch.cat((x, x2), axis=1)
        x = self.relu(self.conv5(x))  # b,20,161,161
        output = self.conv6(x).unsqueeze(2)  # b,20,161,161 -> b,20,1,161,161

        return output




# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # 39*2=78 -> 32
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn3 = nn.BatchNorm2d(32)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=False, padding_mode='replicate')
#         self.bn4 = nn.BatchNorm2d(32)
#         self.relu4 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
#         self.conv5 = nn.Conv2d(32, 32, kernel_size=1, bias=False, padding_mode='replicate')
#         self.bn5 = nn.BatchNorm2d(32)
#         self.relu5 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x1 = self.conv2(x)
#         x = self.relu3(self.bn3(self.conv3(self.upsample(x1))))
#         x = self.avgpool(self.relu4(self.bn4(self.conv4(x))))
#         x = x + x1
#         x = self.relu5(self.bn5(self.conv5(x)))
#
#         return x
#
#
# class SubModel(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 32 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True, padding_mode='replicate')
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(x)))
#         x = self.conv3(x)
#         return x
#
#
# class SubModel4(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 32 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True, padding_mode='replicate')
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(x)))
#         x = self.conv3(x)
#         return x
#
#
#
# class Model(nn.Module):
#     def __init__(self, in_channel=39*2):
#         super().__init__()
#         self.in_channel = in_channel
#         self.encoder = Encoder(self.in_channel, 32)
#         self.sub_model4 = SubModel4(32,20)
#
#
#
#     def forward(self, x):
#         x = x.view(-1, self.in_channel, 161, 161)   # b,t,c,h,w -> b,t*c,h,w
#         x = self.encoder(x)    # b,32,161,161
#         output = self.sub_model4(x).unsqueeze(2)    # b,32,161,161 -> b,20,1,161,161
#         return output

if __name__ == '__main__':
    model = Model(39*2)
    x = torch.randn((1, 39*2, 161, 161))
    print(model(x).shape)