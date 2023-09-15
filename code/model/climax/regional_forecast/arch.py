# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
from climax.arch import ClimaX

class RegionalClimaX(ClimaX):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, mlp_ratio, drop_path, drop_rate)
        self.default_vars = default_vars

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):    # region_info
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        # var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + self.var_embed.unsqueeze(2)  # B, V, L, D

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, lead_times, variables):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        x = F.interpolate(x, size=(160, 160), mode="bilinear")
        out_transformers = self.forward_encoder(x, lead_times, variables)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        preds = self.unpatchify(preds)[:,-5:]
        preds = F.interpolate(preds, size=(161, 161), mode="bilinear")
        # out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        # preds = preds[:, out_var_ids]

        return preds



if __name__ == '__main__':
    ...
    # import xarray as xr
    #
    # data = xr.open_zarr('../../data/weather_round1_train_2011', consolidated=True).x
    #
    # target_features = {'t2m': ['z150', 'z200', 'z250', 'z300', 'z400', 'z500', 't100', 't150', 't250',
    #                            't300', 't400', 't500', 't600', 't700', 'u50', 'r150', 't2m'],  # >0.9
    #                    'u10': ['t100', 't150', 't700', 'u50', 'u850', 'u925', 'u1000', 'r50', 'r100', 't2m', 'u10'],
    #                    # >0.75
    #                    'v10': ['v1000', 'v10', 'v925', 'u1000', 'u10', 'u925'],  # 均值相关的只有三个，加上了方差相关的
    #                    'msl': ['z1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'r50',
    #                            'r100', 'r150', 'r200', 'msl', 'tp'],  # >0.8
    #                    'tp': ['t100', 't150', 't250', 't300', 't400', 't500', 'u50', 'r50', 'r100',
    #                           'r150', 'r200', 'r850', 'r925', 'r1000', 'msl', 'tp']  # >0.8
    #                    }
    # # 把字典的所有值合并成一个没有重复元素的集合
    # all_features = list(set(sum(target_features.values(), [])))
    # target = ['t2m', 'u10', 'v10', 'msl', 'tp']
    # for i in target:
    #     all_features.remove(i)
    #     all_features.append(i)
    #
    # data = data.sel(channel=all_features)
    # print(data.shape)
    # data = torch.from_numpy(data.values).float()
    # torch.save(data, '../../data/2011.pt')


    # all_features.remove(['t2m', 'u10', 'v10', 'msl', 'tp'])
    # print(len(all_features), '\n', all_features)
    # all_features.extend(['t2m', 'u10', 'v10', 'msl', 'tp'])
    # print(len(all_features), '\n', all_features)

    # default_vars = {}
    # img_size = (160, 160)
    # patch_size = 8



    model = RegionalClimaX(default_vars=['ts', 't2m', 'tp', 'w10', 'mslp', '123'], img_size=(160, 160), patch_size=8,
                           embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1).cuda()
    x = torch.randn(3, 6, 161, 161).cuda()
    lead_times = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
    variables = ['ts', 't2m', 'tp', 'w10', 'mslp', '123']
    # out_variables = ['t2m', 'tp', 'w10', 'mslp']
    # for i in range(2):
    preds = model(x, lead_times, variables)
    print(preds.shape)
    with torch.no_grad():
        print(model(x, lead_times, variables).shape)