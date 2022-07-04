import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from modules.ds_modules import GCNNet, LayerIn2LayerOut, LayerOut2LayerIn


class DS_EmbeddingGenerator(nn.Module):
    def __init__(self, surf_num: int, feature_dim: int = 20, gcn_hidden_dim: int = 32, linear_hidden_dim: int = 64,
                 embedding_dim: int = 32):
        super(DS_EmbeddingGenerator, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.LayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.LayerGCN.append(GCNNet(in_dim=feature_dim, hidden_dim=gcn_hidden_dim, out_dim=embedding_dim))
        self.linear1 = nn.Linear(embedding_dim, linear_hidden_dim)
        self.linear2 = nn.Linear(linear_hidden_dim, embedding_dim)

    def forward(self, ds_list: list):
        # DS Layer MessagePassing
        last_layer = None
        for i in range(len(ds_list)):
            x = ds_list[i].x.float().to(self.device)
            edge_index = ds_list[i].edge_index.long().to(self.device)
            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)
            layer_feature = self.LayerGCN[i](x=x, edge_index=edge_index)
            degrees = ds_list[i].degrees.to(self.device)

            sum_list = [[layer_feature[_]] for _ in range(len(x))]
            if len(degrees) != 0:
                for recv, src in zip(layer_edge_index[0], layer_edge_index[1]):
                    sum_list[recv].append(last_layer[src])
                last_layer = [sum(node_message)/degrees[i] for node_message in sum_list]
            else:
                last_layer = layer_feature
        # Forward NN
        for i in range(len(last_layer)):
            last_layer[i] = F.relu(self.linear2(F.relu(self.linear1(last_layer[i]))))

        return torch.stack(last_layer).to(self.device)


class DS_CycleEmbdGenerator(nn.Module):
    def __init__(self, surf_num: int, feature_dim: int = 20, gcn_hidden_dim: int = 32, linear_hidden_dim: int = 64,
                 embedding_dim: int = 32):
        super(DS_CycleEmbdGenerator, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.surf_num = surf_num
        self.InOut = LayerIn2LayerOut()
        self.OutIn = LayerOut2LayerIn()
        self.LayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.LayerGCN.append(GCNConv(in_channels=feature_dim, out_channels=embedding_dim))
        self.OutLayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.OutLayerGCN.append(GCNConv(in_channels=embedding_dim, out_channels=embedding_dim))

    def forward(self, ds_list: list):
        # DS Layer MessagePassing
        layers = [ds_list[i].x.float().to(self.device) for i in range(self.surf_num)]
        layer_edge_index = None
        # in step
        for i in range(len(ds_list)-1, -1, -1):
            edge_index = ds_list[i].edge_index.long().to(self.device)

            layers[i] = F.relu(self.LayerGCN[i](x=layers[i], edge_index=edge_index))

            if i < len(ds_list)-1:
                layers[i] = self.OutIn(layers[i], layers[i+1], layer_edge_index)

            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)

        final_layer = []
        for i in range(len(ds_list)):
            x = layers[i]
            edge_index = ds_list[i].edge_index.long().to(self.device)
            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)

            layer_feature = F.relu(self.OutLayerGCN[i](x=x, edge_index=edge_index))

            if i > 0:
                final_layer = self.InOut(layers[i-1], layer_feature, layer_edge_index)
            # Forward NN
            for i in range(len(final_layer)):
                final_layer[i] = F.relu(self.linear2(F.relu(self.linear1(final_layer[i]))))

            return torch.stack(final_layer).to(self.device)
