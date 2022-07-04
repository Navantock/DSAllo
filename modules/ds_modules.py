import torch
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class LayerIn2LayerOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, ds_in, ds_out, layer_edge_index):
        sum_list = [[_] for _ in ds_out]
        for recv, src in zip(layer_edge_index[0], layer_edge_index[1]):
            sum_list[recv].append(ds_in[src])
        new_layer = [sum(node_message) / torch.tensor(len(node_message)).to(self.device)
                     for node_message in sum_list]
        return torch.stack(new_layer).to(self.device)


class LayerOut2LayerIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, ds_in, ds_out, layer_edge_index):
        sum_list = [[_] for _ in ds_in]
        for src, recv in zip(layer_edge_index[0], layer_edge_index[1]):
            sum_list[recv].append(ds_out[src])
        new_layer = [sum(node_message) / torch.tensor(len(node_message)).to(self.device)
                     for node_message in sum_list]
        return torch.stack(new_layer).to(self.device)
