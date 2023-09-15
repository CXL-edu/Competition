import torch


@torch.no_grad()
def convnet_one_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        x, y = [x.half().cuda(non_blocking=True) for x in batch]

        with torch.cuda.amp.autocast():
            out = model(x)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss

    loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def convnet_pretrain_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        x, y, t = [x.half().cuda(non_blocking=True).view(-1, *x.shape[2:]) for x in batch]

        with torch.cuda.amp.autocast():
            out = model(x, t)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss

    loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def climax_pretrain_evaluate(data_loader, model, criterion, all_features):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        x, y, lead_time = [x.half().cuda() for x in batch]
        x, y = x.view(-1, *x.shape[2:]), y.view(-1, *y.shape[2:])
        lead_time = lead_time.view(-1)

        with torch.cuda.amp.autocast():
            out = model(x, lead_time, all_features)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss

    loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def fourcastnet_pretrain_evaluate(data_loader, model, criterion, ema=None):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    ema.apply_shadow() if ema is not None else None
    for batch in data_loader:
        x, y = [x.half().cuda(non_blocking=True) for x in batch]
        # x = x.transpose(3, 2).transpose(2, 1)
        # y = y.transpose(3, 2).transpose(2, 1)

        with torch.cuda.amp.autocast():
            out = model(x)
            tmp_loss = criterion(out, y)
            if torch.isnan(tmp_loss).int().sum() == 0:
                count += 1
                loss += tmp_loss
    ema.restore() if ema is not None else None
    loss_val = loss.item() / count.item()
    return loss_val


@torch.no_grad()
def fourcastnet_finetune_evaluate(data_loader, model, criterion):
    loss = torch.tensor(0., device="cuda")
    count = torch.tensor(1e-5, device="cuda")

    # switch to evaluation mode
    model.eval()
    for batch in data_loader:
        xt0, xt1, xt2 = [x.half().cuda(non_blocking=True) for x in batch]

        with torch.cuda.amp.autocast():
            out = model(xt0)
            loss += criterion(out, xt1)
            input_2 = torch.cat([xt0[-1], out], dim=0)
            out = model(input_2)
            loss += criterion(out, xt2)
        count += 1

    loss_val = loss.item() / count.item()
    return loss_val


