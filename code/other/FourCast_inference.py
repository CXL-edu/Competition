import os
# import numpy as np
# import xarray as xr
#
# from tqdm import tqdm
from functools import partial

import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
#
from model.afnonet import AFNONet
from utils.params import get_fourcastnet_args
from utils.tools import getModelSize, load_model, save_model

# 把数据一次全部读取，放入dataset和dataloader
data = []
for idx in range(300):
    idx = str(idx).zfill(3)     # 将idx转换成3位数的字符串，在前面补0
    data.append(torch.load(f'./data/weather_round1_test/input/{idx}.pt', map_location='cpu'))

h, w = 162, 162
x_c, y_c = 70, 5
args = get_fourcastnet_args()
model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)).cuda()
ckpt = torch.load('checkpoint/backbone.pt', map_location="cpu")
model.load_state_dict(ckpt['model'])


if not os.path.exists('output'):
    os.mkdir('output')

# inference
for i in range(300):
    output = []
    idx = str(i).zfill(3)
    data_in = data[i].cuda().unsqueeze(0)
    # for j in range(20):
    #     data_out = model(data_in)
    #     output.append(data_out.cpu().detach().numpy())
    #     data_in = torch.cat((data_in[:, -1:], data_out.unsqueeze(1)), dim=1)
    # output = np.concatenate(output, axis=0)
    output = model(data_in).cpu().detach()
    print(output.shape, type(output), output.dtype)
    # output = torch.tensor(output, dtype=torch.float16)
    torch.save(output, f'./output/{idx}.pt')
    print(f'save {idx}.pt ...')
    break


