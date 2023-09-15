import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channel=70 * 2):
        super().__init__()
        self.in_channel = in_channel
        # Encoder 70*2=140 -> 20
        self.conv1 = nn.Conv2d(in_channel, 124, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 140, 161, 161] -> [B, 124, 161, 161
        self.bn1 = nn.BatchNorm2d(124)
        self.conv2 = nn.Conv2d(124, 64, kernel_size=5, padding=2, bias=True, padding_mode='replicate')  # [B, 124, 161, 161] -> [B, 64, 161, 161]
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 64, 161, 161] -> [B, 64, 322, 322]
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')  # [B, 64, 322, 322] -> [B, 32, 322, 322]
        self.conv3_2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2, padding_mode='replicate')  # [B, 64, 322, 322] -> [B, 32, 161, 161]
        self.conv3_3 = nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=2, bias=False, padding_mode='replicate') # [B, 64, 322, 322] -> [B, 32, 161, 161]
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv4_2 = nn.Conv2d(32, 32, kernel_size=5, padding=4, dilation=2, bias=False, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv4_3 = nn.Conv2d(32, 20, kernel_size=7, padding=3, bias=False, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 20, 161, 161]
        self.bn4 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]
        self.conv6 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=False, padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_channel, 161, 161)  # b,t,c,h,w -> b,t*c,h,w
        x = self.relu(self.bn1(self.conv1(x)))  # b,124,161,161
        x = self.relu(self.bn2(self.conv2(x)))  # b,64,161,161
        x = self.upsample(x)  # b,64,161,161 -> b,64,322,322
        x = self.relu(self.conv3_1(x))  # b,64,322,322 -> b,32,322,322
        x = self.relu(self.conv3_2(x))  # b,32,322,322 -> b,32,161,161
        x = self.relu(self.conv3_3(x))  # b,32,322,322 -> b,32,161,161
        x = self.relu(self.bn3(self.conv4_1(x)))  # b,32,161,161
        x = self.relu(self.conv4_2(x))  # b,32,161,161
        x = self.relu(self.bn4(self.conv4_3(x)))  # b,20,161,161
        # x = torch.cat((x, x2), axis=1)
        x = self.conv5(x)  # b,20,161,161
        output = self.conv6(x).unsqueeze(2)  # b,20,161,161 -> b,20,1,161,161


        return output


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 2, 70, 161, 161)
    y = model(x)
    print(y.shape)