# import pickle
# from pathlib import Path
#
# import os
# import numpy as np
# from ffrecord.ffrecord import FileReader

import xarray as xr
# import torch
# from torch_geometric.data import Data
from torch.utils.data import Dataset
# import data_factory.graph_tools as gg


# class StandardScaler:
#     def __init__(self):
#         self.mean = 0.0
#         self.std = 1.0
#
#     def load(self, scaler_dir):
#         with open(scaler_dir, "rb") as f:
#             pkl = pickle.load(f)
#             self.mean = pkl["mean"]
#             self.std = pkl["std"]
#
#     def inverse_transform(self, data):
#         mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
#         std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
#         return (data * std) + mean


class ERA5(Dataset):

    def __init__(self, data: 'xarray.DataArray', modelname: str = 'fourcastnet') -> None:

        assert modelname in ["fourcastnet", "graphcast"]
        self.modelname = modelname

        self.data = data

        self.constant_features = None

    def __len__(self):
        return self.data.shape[0] - 3 + 1

    def __getitem__(self, indices):
        samples = []
        # if self.modelname == 'fourcastnet':
        x0 = self.data[indices].values
        x1 = self.data[indices+1].values
        y = self.data[indices+2].values
        # x0 = np.nan_to_num(x0[:, :, :-2])
        # x1 = np.nan_to_num(x1[:, :, :-2])
        # y = np.nan_to_num(y[:, :, :-2])
        samples.append((x0, x1, y))
        # else:
        #     x = np.nan_to_num(np.reshape(np.concatenate([x0, x1, y[:, :, -2:]], axis=-1), [-1, 49]))
        #     y = np.nan_to_num(np.reshape(y[:, :, :-2], [-1, 20]))
        #     samples.append((x, y))
        return samples



# class EarthGraph(object):
#     def __init__(self):
#         self.mesh_data = None
#         self.grid2mesh_data = None
#         self.mesh2grid_data = None
#
#     def generate_graph(self):
#         mesh_nodes = gg.fetch_mesh_nodes()
#
#         mesh_6_edges, mesh_6_edges_attrs = gg.fetch_mesh_edges(6)
#         mesh_5_edges, mesh_5_edges_attrs = gg.fetch_mesh_edges(5)
#         mesh_4_edges, mesh_4_edges_attrs = gg.fetch_mesh_edges(4)
#         mesh_3_edges, mesh_3_edges_attrs = gg.fetch_mesh_edges(3)
#         mesh_2_edges, mesh_2_edges_attrs = gg.fetch_mesh_edges(2)
#         mesh_1_edges, mesh_1_edges_attrs = gg.fetch_mesh_edges(1)
#         mesh_0_edges, mesh_0_edges_attrs = gg.fetch_mesh_edges(0)
#
#         mesh_edges = mesh_6_edges + mesh_5_edges + mesh_4_edges + mesh_3_edges + mesh_2_edges + mesh_1_edges + mesh_0_edges
#         mesh_edges_attrs = mesh_6_edges_attrs + mesh_5_edges_attrs + mesh_4_edges_attrs + mesh_3_edges_attrs + mesh_2_edges_attrs + mesh_1_edges_attrs + mesh_0_edges_attrs
#
#         self.mesh_data = Data(x=torch.tensor(mesh_nodes, dtype=torch.float),
#                               edge_index=torch.tensor(mesh_edges, dtype=torch.long).T.contiguous(),
#                               edge_attr=torch.tensor(mesh_edges_attrs, dtype=torch.float))
#
#         grid2mesh_edges, grid2mesh_edge_attrs = gg.fetch_grid2mesh_edges()
#         self.grid2mesh_data = Data(x=None,
#                                    edge_index=torch.tensor(grid2mesh_edges, dtype=torch.long).T.contiguous(),
#                                    edge_attr=torch.tensor(grid2mesh_edge_attrs, dtype=torch.float))
#
#         mesh2grid_edges, mesh2grid_edge_attrs = gg.fetch_mesh2grid_edges()
#         self.mesh2grid_data = Data(x=None,
#                                    edge_index=torch.tensor(mesh2grid_edges, dtype=torch.long).T.contiguous(),
#                                    edge_attr=torch.tensor(mesh2grid_edge_attrs, dtype=torch.float))



if __name__ == '__main__':
    data = xr.open_zarr('../data/weather_round1_train_2011', consolidated=True).x
    print(data.shape)
    data = ERA5(data)
    for temp in data:
        print(temp[0][0].shape)
        break

