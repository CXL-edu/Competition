import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: 时间编码（只能相对时间？测试集中好像没有绝对时间）
# TODO：自回归微调

class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x ,lead_times):
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

        self.conv3d = nn.Conv3d(70, 70, kernel_size=(2, 1, 1), padding=0)  # [B, 70, 2, 161, 161] -> [B, 70, 1, 161, 161]

        # encode
        self.conv0 = nn.Conv2d(ch[0], ch[1], 4, 1, 1)  # [B, 70, 161, 161] -> [B, 64, 160, 160]
        self.conv1 = nn.Conv2d(ch[1], ch[1], 5, 2, padding=2)  # [B, 64, 160, 160] -> [B, 64, 80, 80]
        self.conv2 = nn.Conv2d(ch[1], ch[2], 3, 1, padding=1)  # [B, 64, 80, 80] -> [B, 32, 80, 80]
        self.conv3 = nn.Conv2d(ch[2], ch[2], 5, 2, padding=2)  # [B, 32, 80, 80] -> [B, 32, 40, 40]
        self.conv4 = nn.Conv2d(ch[2], ch[3], 3, 1, padding=1)  # [B, 32, 40, 40] -> [B, 16, 40, 40]
        self.conv5 = nn.Conv2d(ch[3], ch[3], 5, 2, padding=2)  # [B, 16, 40, 40] -> [B, 16, 20, 20]
        self.conv6 = nn.Conv2d(ch[3], ch[4], 3, 1, padding=1)  # [B, 16, 20, 20] -> [B, 8, 20, 20]

        # decode
        # self.conv6 = nn.ConvTranspose2d(ch[2], ch[2], 5, 2, padding=1)  # [B, 32, 20, 20] -> [B, 32, 40, 40]
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 8, 20, 20] -> [B, 8, 40, 40]
        self.conv1_d = nn.Conv2d(ch[4]+ch[3], ch[3], 3, 1, padding=1)  # cat(([B, 8, 40, 40], self.conv4(x)), axis=1) -> [B, 16, 40, 40]
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 16, 40, 40] -> [B, 16, 80, 80]
        self.conv2_d = nn.Conv2d(ch[3]+ch[2], 48, 3, 1, padding=1)  # cat(([B, 16, 80, 80], self.conv2(x)), axis=1) -> [B, 24, 80, 80]
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)  # [B, 24, 80, 80] -> [B, 24, 160, 160]
        self.conv3_d = nn.Conv2d(48+ch[1], ch[1], 3, 1, padding=1)  # cat(([B, 24, 160, 160], self.conv1(x)), axis=1) -> [B, 32, 160, 160]
        self.conv4_d = nn.Conv2d(ch[1], ch[2], 5, 1, padding=2)  # [B, 32, 160, 160] -> [B, 5, 160, 160]
        self.conv5_d = nn.Conv2d(ch[2], ch[3], 5, 1, padding=2)  # [B, 32, 160, 160] -> [B, 16, 160, 160]
        # 插值到161*161
        self.conv6_d = nn.Conv2d(ch[3]+1, 12, 5, 1, padding=2)  # cat([B, 16, 161, 161], time_emb) -> [B, 5, 161, 161]
        self.conv7_d = nn.Conv2d(12, 5, 5, 1, padding=2)  # cat([B, 16, 161, 161], time_emb) -> [B, 5, 161, 161]

        self.relu = nn.ReLU()
        self.time_embed = nn.Embedding(20, 161*161)
        # self.encoder = Encoder(2 * 70, 5, 128, 2, 0.1)
        # self.decoder = Decoder()
        # self.time_embed = nn.Embedding(5, 128)
        # self.pos_embed = nn.Parameter(torch.randn(1, 1, 161, 161))

    def forward(self, x, time):   # , lead_times
        # x = self.conv3d(x.transpose(1, 2)).squeeze(2)  # [B, 2, 70, 161, 161] -> [B, 70, 161, 161]
        x = x.view(x.shape[0], 70*2, 161, 161) # [B, 2, 70, 161, 161] -> [B, 70*2, 161, 161]

        x_0 = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x_0))
        x_2 = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x_2))
        x_4 = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x_4))
        x = self.relu(self.conv6(x))

        x = self.upsample1(x)
        x = torch.cat([x, x_4], dim=1)
        x = self.relu(self.conv1_d(x))
        x = self.upsample2(x)
        x = torch.cat([x, x_2], dim=1)
        x = self.relu(self.conv2_d(x))
        x = self.upsample3(x)
        x = torch.cat([x, x_0], dim=1)
        x = self.relu(self.conv3_d(x))
        x = self.relu(self.conv4_d(x))
        x = self.relu(self.conv5_d(x))
        x = F.interpolate(x, size=(161, 161), mode="bicubic")
        # print(f'time:{time.shape}')
        time_emb = self.time_embed(time).view(x.shape[0], 1, 161, 161)
        # print(f'time_emb:{time_emb.shape}')
        # time_emb = time_emb.view(x.shape[0], 1, 161, 161)
        x = torch.cat([x, time_emb], dim=1)
        x = self.relu(self.conv6_d(x))
        x = self.relu(self.conv7_d(x))
        return x


if __name__ == '__main__':
    # import xarray as xr
    # data = torch.load('../data/weather_round1_test/input/000.pt').unsqueeze(0)
    # print(data.shape)

    # torch.Size([160, 2, 70, 161, 161])
    # torch.Size([160, 5, 161, 161])
    # torch.Size([160, 1])

    x = torch.randn(160, 2, 70, 161, 161)
    y = torch.randn(160, 5, 161, 161)
    time = torch.arange(0,20).repeat(8,1).view(-1,1)

    model = Model()
    preds = model(x, time)
    print(preds.shape)
    criterion = nn.MSELoss()
    loss = criterion(preds, y)
    print(loss)

    # print(model(data, torch.tensor([13])).shape)