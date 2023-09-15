import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channel=39 * 2):
        super().__init__()
        self.in_channel = in_channel
        # Encoder 39*2=78 -> 32
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, padding=3, bias=False, padding_mode='replicate')  # [B, 78, 161, 161] -> [B, 64, 161, 161
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 64, 161, 161]
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=False, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv4 = nn.Conv2d(32, 20, kernel_size=7, padding=3, bias=False, padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 20, 161, 161]
        self.bn3 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=False, padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]
        self.conv6 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=False, padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]

        self.register_parameter('time_embedding', nn.Parameter(torch.randn(1, 20, 161, 161)))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.in_channel, 161, 161)  # b,t,c,h,w -> b,t*c,h,w
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
        time_embedding = self.time_embedding.repeat(x.shape[0], 1, 1, 1)
        x = x + time_embedding
        x = self.relu(self.conv5(x))  # b,20,161,161
        output = self.conv6(x).unsqueeze(2)  # b,20,161,161 -> b,20,1,161,161

        return output


if __name__ == '__main__':
    model = Model()
    x = torch.randn(1, 2, 39, 161, 161)
    y = model(x)
    print(y.shape)

    for key, value in model.named_parameters():
        print(key, value.shape)
    print(model)