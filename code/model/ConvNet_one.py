import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: 时间编码（只能相对时间？测试集中好像没有绝对时间）
# TODO：自回归微调

class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        raise NotImplementedError


# class Encoder(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         channels = [70, 64, 32, 16, 8]
#
#
#     def forward(self, x):
#         # x: [B, L, D]
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x


class Model(ModelBase):
    def __init__(self):
        super().__init__()
        # ([B, 2*70, H, W], t) -> [B, 5, H, W]
        # ([B, 2*70, 161, 161], t) -> [B, 5, 161, 161]
        ch = [70, 64, 32, 16, 8]
        ch = [x * 2 for x in ch]

        # encode
        self.conv0 = nn.Conv2d(140, 140, 4, 2, 1)  # [B, 140, 161, 161] -> [B, 140, 80, 80]
        self.conv1 = nn.Conv2d(140, 128, 1, 1)  # [B, 128, 80, 80] -> [B, 128, 80, 80]
        self.ln1 = nn.LayerNorm([128, 80, 80])
        self.conv2 = nn.Conv2d(128, 128, 5, 1, padding=2)   # [B, 128, 80, 80] -> [B, 128, 80, 80]
        # self.conv2 = nn.Conv2d(128, 128, 7, 2, padding=3)  # [B, 128, 80, 80] -> [B, 128, 40, 40]
        self.conv3 = nn.Conv2d(128, 64, 1, 1)  # [B, 128, 80, 80] -> [B, 64, 80, 80]
        self.ln2 = nn.LayerNorm([64, 80, 80])
        self.conv4 = nn.Conv2d(64, 64, 5, 1, padding=2)  # [B, 64, 80, 80] -> [B, 64, 80, 80]
        self.conv5 = nn.Conv2d(64, 64, 7, 2, 3)  # [B, 64, 80, 80] -> [B, 64, 40, 40]
        self.conv6 = nn.Conv2d(64, 32, 1, 1)  # [B, 64, 40, 40] -> [B, 32, 40, 40]
        self.ln3 = nn.LayerNorm([32, 40, 40])
        self.conv7 = nn.Conv2d(32, 32, 5, 1, padding=2)  # [B, 32, 40, 40] -> [B, 32, 40, 40]

        # decode
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 32, 40, 40] -> [B, 32, 80, 80]
        self.conv1_d = nn.Conv2d(32+64, 64, 5, 1, padding=2)  # cat([B, 32, 80, 80], conv4(x)) -> [B, 64, 80, 80]
        self.ln1_d = nn.LayerNorm([64, 80, 80])
        self.conv2_d = nn.Conv2d(64+64, 128, 5, 1, padding=2)  # cat([B, 64, 80, 80], conv3(x)) -> [B, 128, 80, 80]
        self.ln2_d = nn.LayerNorm([128, 80, 80])
        self.conv3_d = nn.Conv2d(128+128, 128, 5, 1, padding=2)  # cat([B, 128, 80, 80], conv1(x)) -> [B, 128, 80, 80]
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 128, 80, 80] -> [B, 128, 160, 160]
        # 将图片插值到161*161
        self.conv4_d = nn.Conv2d(128, 100, 5, 1, padding=2)  # cat([B, 128, 161, 161], conv1(x)) -> [B, 100, 161, 161]

        self.relu = nn.ReLU()


    def forward(self, x):   # , lead_times
        # x: [B, 2*70, 161, 161] -> [B, 20*5, 161, 161] -> [B, 20, 5, 161, 161]
        x = x.view(x.shape[0], -1, 161, 161)
        x = self.relu(self.conv0(x))
        x1 = self.conv1(x)
        x = self.relu(self.ln1(x1))
        x = self.relu(self.conv2(x))
        x3 = self.conv3(x)
        x = self.relu(self.ln2(x3))
        x4 = self.conv4(x)
        x = self.relu(x4)
        x = self.relu(self.conv5(x))
        x = self.relu(self.ln3(self.conv6(x)))
        x = self.relu(self.conv7(x))

        x = self.upsample1(x)
        # print(x.shape, x5.shape)
        x = self.conv1_d(torch.cat([x, x4], dim=1))
        x = self.relu(self.ln1_d(x))
        x = self.conv2_d(torch.cat([x, x3], dim=1))
        x = self.relu(self.ln2_d(x))
        x = self.conv3_d(torch.cat([x, x1], dim=1))
        x = self.upsample2(x)
        x = F.interpolate(x, size=(161, 161), mode="bicubic", align_corners=True)
        x = self.conv4_d(x)
        x = x.view(x.shape[0], 20, 5, 161, 161)

        return x


if __name__ == '__main__':
    # import xarray as xr
    # data = torch.load('../data/weather_round1_test/input/000.pt').unsqueeze(0)
    # print(data.shape)

    # torch.Size([160, 2, 70, 161, 161])
    # torch.Size([160, 5, 161, 161])
    # torch.Size([160, 1])

    x = torch.randn(160, 2, 70, 161, 161)
    y = torch.randn(160, 20, 5, 161, 161)
    # time = torch.arange(0,20).repeat(8,1).view(-1,1)

    model = Model()
    preds = model(x)
    print(preds.shape)
    criterion = nn.MSELoss()
    loss = criterion(preds, y)
    print(loss)

    # print(model(data, torch.tensor([13])).shape)