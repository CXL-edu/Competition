import os
import time
from pathlib import Path
import numpy as np
import xarray as xr

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
from timm.scheduler import create_scheduler

from model.afnonet import AFNONet
from utils.params import get_fourcastnet_args
from utils.tools import getModelSize, load_model, save_model
from utils.eval import fourcastnet_pretrain_evaluate, fourcastnet_finetune_evaluate

SAVE_PATH = Path('./checkpoint/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


class ERA5(Dataset):

    def __init__(self, data: 'xarray.DataArray', modelname: str = 'fourcastnet') -> None:

        assert modelname in ["fourcastnet", "graphcast"]
        self.modelname = modelname
        self.data = data

    def __len__(self):
        return self.data.shape[0] - 4 + 1

    def __getitem__(self, idx):
        # samples = []
        # if self.modelname == 'fourcastnet':
        x0 = torch.from_numpy(self.data[idx:idx+2].values).float()
        x1 = torch.from_numpy(self.data[idx+2].values[-5:]).float()
        y = torch.from_numpy(self.data[idx+3].values[-5:]).float()
        # print(type(x0), x0.shape, x0.dtype)
        # x0 = np.nan_to_num(x0[:, :, :-2])
        # x1 = np.nan_to_num(x1[:, :, :-2])
        # y = np.nan_to_num(y[:, :, :-2])
        # samples.append((x0, x1, y))

        return x0, x1, y



def pretrain_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss):
    loss_val = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    model.train()

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue

        x, y, _ = [x.half().cuda(non_blocking=True) for x in batch]
        # xt0 = xt0.permute(3, 1, 2)
        # x = x.transpose(3, 2).transpose(2, 1)
        # y = y.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss).int().sum() == 0:
                count += 1
                loss_val += loss

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        save_model(model, epoch, step+1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH/'pretrain_latest.pt')

    return loss_val.item() / count.item()


def finetune_one_epoch(epoch, start_step, model, criterion, data_loader, optimizer, loss_scaler, lr_scheduler, min_loss):
    loss_val = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    model.train()

    for step, batch in enumerate(data_loader):
        if step < start_step:
            continue

        xt0, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]
        # xt0 = xt0.permute(3, 1, 2)
        # xt0 = xt0.transpose(3, 2).transpose(2, 1)
        # xt1 = xt1.transpose(3, 2).transpose(2, 1)
        # xt2 = xt2.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(xt0)
            loss = criterion(out, xt1)
            # 把原始输入的第二部分与输出拼接起来，作为下一步的输入
            input_2 = torch.cat([xt0[:, -1:], out.unsqueeze(1)], dim=1)
            out = model(input_2)
            loss += criterion(out, xt2)
            if torch.isnan(loss).int().sum() == 0:
                count += 1
                loss_val += loss

        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()
        optimizer.zero_grad()

        save_model(model.module, epoch, step + 1, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'finetune_latest.pt')
    if loss_val.item() < min_loss:
        min_loss = loss_val.item()
        save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)

    return loss_val.item() / count.item()


def train(args):
    # input size
    h, w = 162, 162
    x_c, y_c = 70, 5

    model = AFNONet(img_size=[h, w], in_chans=x_c, out_chans=y_c, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)).cuda()
    param_sum, buffer_sum, all_size = getModelSize(model)
    print(f"Number of Parameters: {param_sum}, Number of Buffers: {buffer_sum}, Size of Model: {all_size:.4f} MB")
    print(args)
    # param_groups = timm.optim.optim_factory.add_weight_decay(model, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    loss_scaler = torch.cuda.amp.GradScaler(enabled=True)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.MSELoss()

    # load data
    data = xr.open_zarr('./data/weather_round1_train_2011', consolidated=True).x
    data_train, data_val = data[:-200], data[-200:]  # 划分训练集和验证集 (sample, channel, height, width)
    train_dataset = ERA5(data_train)
    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=8, pin_memory=True, drop_last=False)  # , num_workers=8, pin_memory=True
    val_dataset = ERA5(data_val)
    val_dataloader = DataLoader(val_dataset, args.batch_size, num_workers=8, pin_memory=True, drop_last=False)

    # load
    start_epoch, start_step, min_loss = load_model(model, optimizer, lr_scheduler, loss_scaler, SAVE_PATH / 'pretrain_latest.pt')
    # print(torch.load(SAVE_PATH / 'backbone.pt').items())
    # model.load_state_dict(torch.load(SAVE_PATH / 'backbone.pt')['model'])
    # start_epoch, start_step = 1, 1
    # min_loss = np.inf
    print(f"Start pretrain for {args.pretrain_epochs} epochs")

    for epoch in tqdm(range(start_epoch, args.pretrain_epochs)):
        t0 = time.time()
        train_loss = pretrain_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        t1 = time.time()
        start_step = 0
        lr_scheduler.step(epoch)

        val_loss = fourcastnet_pretrain_evaluate(val_dataloader, model, criterion)

        # if rank == 0 and local_rank == 0:
        print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, Time: {t1-t0:.2f}s")
        if val_loss < min_loss:
            min_loss = val_loss
            save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'backbone_latest.pt')
        save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'pretrain_latest.pt')


    # load
    start_epoch, start_step, min_loss = load_model(model, optimizer, lr_scheduler, loss_scaler, SAVE_PATH / 'finetune_latest.pt')
    # if local_rank == 0:
    print(f"Start finetune for {args.finetune_epochs} epochs")

    for epoch in tqdm(range(start_epoch, args.finetune_epochs)):

        train_loss = finetune_one_epoch(epoch, start_step, model, criterion, train_dataloader, optimizer, loss_scaler, lr_scheduler, min_loss)
        start_step = 0
        lr_scheduler.step(epoch)

        val_loss = fourcastnet_finetune_evaluate(val_dataloader, model, criterion)

        # if rank == 0 and local_rank == 0:
        print(f"Epoch {epoch} | Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
        if val_loss < min_loss:
            min_loss = val_loss
            save_model(model, path=SAVE_PATH / 'backbone.pt', only_model=True)
            save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'finetune_backbone_latest.pt')
        save_model(model, epoch + 1, 0, optimizer, lr_scheduler, loss_scaler, min_loss, SAVE_PATH / 'finetune_latest.pt')



def main(args):
    # fix the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    train(args)




if __name__ == '__main__':
    args = get_fourcastnet_args()
    main(args)