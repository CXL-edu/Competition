import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channel=39 * 2):
        super().__init__()
        self.in_channel = in_channel
        # Encoder 39*2=78 -> 32
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, padding=3, bias=False,
                               padding_mode='replicate')  # [B, 78, 161, 161] -> [B, 64, 161, 161
        # self.gn1 = nn.GroupNorm(4, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=True,
                                 padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 64, 161, 161]
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2, bias=False,
                               padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True,
                                 padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv3 = nn.Conv2d(32, 32, kernel_size=7, padding=3, bias=False,
                               padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 32, 161, 161]
        self.conv4 = nn.Conv2d(32, 20, kernel_size=7, padding=3, bias=False,
                               padding_mode='replicate')  # [B, 32, 161, 161] -> [B, 20, 161, 161]
        self.bn3 = nn.BatchNorm2d(20)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=3, padding=1, bias=False,
                               padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]
        self.conv6 = nn.Conv2d(20, 20, kernel_size=5, padding=2, bias=False,
                               padding_mode='replicate')  # [B, 20, 161, 161] -> [B, 20, 161, 161]

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
        x = self.relu(self.conv5(x))  # b,20,161,161
        output = self.conv6(x)  # b,20,161,161 -> b,20,161,161

        return output

class Decoder(nn.Module):
    def __init__(self, in_channel=20+5+5):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, 12, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 78, 161, 161] -> [B, 64, 161, 161
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2, bias=False, padding_mode='replicate')  # [B, 64, 161, 161] -> [B, 32, 161, 161]


        self.relu = nn.ReLU()

    def forward(self, x, dec_in, time_emb):
        # x: [B, C, H, W]
        # dec_in: [B, 2, H, W]
        # time_emb: [B, 1, H, W]
        x = torch.cat((x, dec_in, time_emb), dim=1) # [B, C+2+1, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return x


class OutModule(nn.Module):
    def __init__(self, in_channel=12):
        super().__init__()
        # [B, 6, 161, 161] -> [B, 2, 161, 161]
        self.conv1 = nn.Conv2d(in_channel, 5, kernel_size=5, padding=2, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.conv3 = nn.Conv2d(5, 5, kernel_size=5, padding=2, bias=False, padding_mode='replicate')

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.conv3(self.conv2(x))
        return x


class MemoryModule(nn.Module):
    def __init__(self, in_channel=5):
        super().__init__()
        # [B, 6, 161, 161] -> [B, 2, 161, 161]
        self.conv1 = nn.Conv2d(in_channel, 5, kernel_size=3, padding=1, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5, padding=2, bias=False, padding_mode='replicate')

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        return x


class Model(nn.Module):
    def __init__(self, in_channel=39*2):
        super().__init__()
        self.in_channel = in_channel
        # Encoder 39*2=78 -> 32
        self.encoder = Encoder(in_channel)
        self.decoder = Decoder(20+5+5)

        self.out_module = OutModule(12)
        # self.memory_module = MemoryModule(5)


        self.time_embedding = nn.Embedding(20, 161*161)
        self.dec_init = nn.Parameter(torch.randn(1, 20, 161, 161))   # 记忆模块的输出维度与其相同，相当于初始记忆



        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.view(-1, self.in_channel, 161, 161)  # b,t,c,h,w -> b,t*c,h,w
        # x = x.transpose(1, 2)  # b,t,c,h,w -> b,c,t,h,w
        enc_out = self.encoder(x) # b,c,t,h,w -> b,c,h,w
        dec_in = self.dec_init.expand(x.shape[0], -1, -1, -1).to(x.device) # 1,c,h,w -> b,c,h,w
        time_emb_all = self.time_embedding(torch.arange(0, 20).to(x.device)).view(1, 20, 161, 161).expand(x.shape[0], -1, -1, -1) # 1,20,h,w -> b,20,h,w

        dec_out = self.decoder(enc_out, dec_in, time_emb_all)  # b,c,h,w -> b,c,h,w
        output = [self.out_module(dec_out)]
        for i in torch.arange(1,4):
            # dec_in = self.memory_module(output[-1]) # b,c,h,w -> b,c,h,w
            dec_in = output[-1]
            time_emb = time_emb_all[:,5*i:5*(i+1)] if i<3 else time_emb_all[:,5*i:]
            dec_out = self.decoder(enc_out, dec_in, time_emb) # b,c,h,w -> b,c,h,w
            output.append(self.out_module(dec_out)) # b,c,h,w -> b,c,1,h,w

        output = torch.cat(output, dim=1).unsqueeze(2) # b,1,h,w -> b,20,1,h,w

        return output


if __name__ == '__main__':
    model = Model().cuda()
    x = torch.randn(3, 2, 39, 161, 161).cuda()
    y = model(x)
    print(y.shape)