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



class Model(nn.Module):
    def __init__(self, features_name):
        super().__init__()

        self.features_name = features_name
        self.feature_len = [len(v) for v in features_name.values()]
        all_features = list(set(sum(features_name.values(), [])))
        self.in_channel = len(all_features)*2

        self.encoder = Encoder(self.in_channel, 32)

        self.sub_model0 = SubModel(32,20)
        self.sub_model1 = SubModel(32,20)
        self.sub_model2 = SubModel(32,20)
        self.sub_model3 = SubModel(32,20)
        self.sub_model4 = SubModel4(32,20)



    def forward(self, x):
        x = x.view(-1, self.in_channel, 161, 161)   # b,t,c,h,w -> b,t*c,h,w
        x = self.encoder(x)    # b,32,161,161
        output0 = self.sub_model0(x).unsqueeze(2)    # b,32,161,161 -> b,20,1,161,161
        output1 = self.sub_model1(x).unsqueeze(2)
        output2 = self.sub_model2(x).unsqueeze(2)
        output3 = self.sub_model3(x).unsqueeze(2)
        output4 = self.sub_model4(x).unsqueeze(2)
        output = torch.cat([output0, output1, output2, output3, output4], dim=2)

        # output0 = self.sub_model0(x0.view(-1, self.feature_len[0]*2, 161, 161)).unsqueeze(2)    # b,t,c,h,w -> b,t*c,h,w -> b,t,1,h,w
        # output1 = self.sub_model1(x1.view(-1, self.feature_len[1]*2, 161, 161)).unsqueeze(2)    # 不使用3D卷积等考虑时空的模型时
        # output2 = self.sub_model2(x2.view(-1, self.feature_len[2]*2, 161, 161)).unsqueeze(2)    # b,2,c,161,161 -> b,20,1,161,161
        # output3 = self.sub_model3(x3.view(-1, self.feature_len[3]*2, 161, 161)).unsqueeze(2)
        # output4 = self.sub_model4(x4.view(-1, self.feature_len[4]*2, 161, 161)).unsqueeze(2)
        # output = torch.cat([output0, output1, output2, output3, output4], dim=2)
        return output


# class Model_V0(nn.Module):
#     def __init__(self, features_name):
#         super().__init__()
#
#         self.features_name = features_name
#         self.feature_len = [len(v) for v in features_name.values()]
#
#         self.sub_model0 = SubModel0(in_channels=self.feature_len[0]*2, out_channels=20)
#         self.sub_model1 = SubModel1(in_channels=self.feature_len[1]*2, out_channels=20)
#         self.sub_model2 = SubModel2(in_channels=self.feature_len[2]*2, out_channels=20)
#         self.sub_model3 = SubModel3(in_channels=self.feature_len[3]*2, out_channels=20)
#         self.sub_model4 = SubModel4(in_channels=self.feature_len[4]*2, out_channels=20)
#
#
#
#     def forward(self, x0, x1, x2, x3, x4):
#
#         output0 = self.sub_model0(x0.view(-1, self.feature_len[0]*2, 161, 161)).unsqueeze(2)    # b,t,c,h,w -> b,t*c,h,w -> b,t,1,h,w
#         output1 = self.sub_model1(x1.view(-1, self.feature_len[1]*2, 161, 161)).unsqueeze(2)    # 不使用3D卷积等考虑时空的模型时
#         output2 = self.sub_model2(x2.view(-1, self.feature_len[2]*2, 161, 161)).unsqueeze(2)    # b,2,c,161,161 -> b,20,1,161,161
#         output3 = self.sub_model3(x3.view(-1, self.feature_len[3]*2, 161, 161)).unsqueeze(2)
#         output4 = self.sub_model4(x4.view(-1, self.feature_len[4]*2, 161, 161)).unsqueeze(2)
#         output = torch.cat([output0, output1, output2, output3, output4], dim=2)
#         return output
#
#
# class SubModel0(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 17*2=34 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(self.upsample(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.conv4(self.avgpool(x))
#         return x
#
#
# class SubModel1(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # channel变化 11*2=22 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 基于双线性插值的上采样
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2) # 平均池化下采样
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(self.upsample(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.conv4(self.avgpool(x))
#         return x
#
#
# class SubModel2(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 6*2=12 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(24)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
#         self.conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(24)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(24, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(self.upsample(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.conv4(self.avgpool(x))
#         return x
#
#
# class SubModel3(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 14*2=28 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(self.upsample(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.conv4(self.avgpool(x))
#         return x
#
#
# class SubModel4(nn.Module):
#     def __init__(self, in_channels, out_channels=20):
#         super().__init__()
#         # 16*2=32 -> 20
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 基于双线性插值的上采样
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)  # 平均池化下采样
#         self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         x = self.relu2(self.bn2(self.conv2(self.upsample(x))))
#         x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.conv4(self.avgpool(x))
#         return x


if __name__ == '__main__':
    model = Model(11)
    x = torch.randn((1, 11, 161, 161))
    print(model(x).shape)