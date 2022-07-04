import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from modules.DS_Embedding import DS_EmbeddingGenerator
from modules.ds_modules import GCNNet, LayerOut2LayerIn, LayerIn2LayerOut


class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim: int = 2, hidden_dim1: int = 32, hidden_dim2: int = 32):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x_embedding):
        x = F.normalize(F.relu(self.linear1(x_embedding)))
        x = F.normalize(F.relu(self.linear2(x)))
        x = self.linear3(x)

        return x


class End2EndPredictor(nn.Module):
    def __init__(self, surf_num: int = 3, feature_dim: int = 20, embedding_dim: int = 32):
        super(End2EndPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.EmbeddingGenerator = DS_EmbeddingGenerator(surf_num=3, feature_dim=feature_dim, embedding_dim=embedding_dim)
        self.Predictor = Predictor(in_dim=embedding_dim)

    def forward(self, ds_list):
        embd = self.EmbeddingGenerator(ds_list)
        probility = self.Predictor(embd)
        return probility


class SimpleLayerGCNPredictor(nn.Module):
    def __init__(self, feature_dim: int = 20, hidden_dim: int = 32, dropout: float = 0.3):
        super(SimpleLayerGCNPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.GCN = GCNNet(in_dim=feature_dim, hidden_dim=hidden_dim, out_dim=2)

    def forward(self, ds_list):
        x = ds_list[-1].x.float().to(self.device)
        edge_index = ds_list[-1].edge_index.long().to(self.device)
        out = F.dropout(self.GCN(x=x, edge_index=edge_index))
        return out


class DS_GCNPredictor(nn.Module):
    def __init__(self, surf_num: int, feature_dim: int = 20, gcn_hidden_dim: int = 32,
                 embedding_dim: int = 32, dp: float = 0.4):
        super(DS_GCNPredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dp = dp
        self.LayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.LayerGCN.append(GCNNet(in_dim=feature_dim, hidden_dim=gcn_hidden_dim, out_dim=embedding_dim))
        self.PredictGCN = GCNNet(in_dim=embedding_dim, hidden_dim=gcn_hidden_dim, out_dim=2)

    def forward(self, ds_list: list):
        # DS Layer MessagePassing
        last_layer = None
        for i in range(len(ds_list)):
            x = ds_list[i].x.float().to(self.device)
            edge_index = ds_list[i].edge_index.long().to(self.device)
            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)
            degrees = ds_list[i].degrees.to(self.device)

            layer_feature = self.LayerGCN[i](x=x, edge_index=edge_index)

            sum_list = [[layer_feature[_]] for _ in range(len(x))]
            if len(degrees) != 0:
                for recv, src in zip(layer_edge_index[0], layer_edge_index[1]):
                    sum_list[recv].append(last_layer[src])
                last_layer = [sum(node_message)/degrees[i] for node_message in sum_list]
            else:
                last_layer = layer_feature

        last_layer = torch.stack(last_layer).to(self.device)
        x_embedding = F.dropout(last_layer, p=self.dp)
        out = self.PredictGCN(x=x_embedding, edge_index=ds_list[-1].edge_index.long().to(self.device))

        return out


class DS_CycleGCNPRedictor(nn.Module):
    def __init__(self, surf_num: int, feature_dim: int = 20, gcn_hidden_dim: int = 32,
                 embedding_dim: int = 32, dp: float = 0.4):
        super(DS_CycleGCNPRedictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dp = dp
        self.surf_num = surf_num
        self.InOut = LayerIn2LayerOut()
        self.OutIn = LayerOut2LayerIn()
        self.LayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.LayerGCN.append(GCNConv(in_channels=feature_dim, out_channels=embedding_dim))
        self.OutLayerGCN = nn.ModuleList()
        for _ in range(surf_num):
            self.OutLayerGCN.append(GCNConv(in_channels=embedding_dim, out_channels=embedding_dim))
        self.PredictGCN = GCNNet(in_dim=embedding_dim, hidden_dim=gcn_hidden_dim, out_dim=2)

    def forward(self, ds_list: list):
        # DS Layer MessagePassing
        layers = [ds_list[i].x.float().to(self.device) for i in range(self.surf_num)]
        layer_edge_index = None
        # in step
        for i in range(len(ds_list) - 1, -1, -1):
            edge_index = ds_list[i].edge_index.long().to(self.device)

            layers[i] = F.relu(self.LayerGCN[i](x=layers[i], edge_index=edge_index))

            if i < len(ds_list) - 1:
                layers[i] = self.OutIn(layers[i], layers[i + 1], layer_edge_index)

            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)

        final_layer = None
        for i in range(len(ds_list)):
            edge_index = ds_list[i].edge_index.long().to(self.device)
            layer_edge_index = ds_list[i].layer_edge_index.to(self.device)

            layer_feature = F.relu(self.OutLayerGCN[i](x=layers[i], edge_index=edge_index))

            if i > 0:
                final_layer = self.InOut(layers[i - 1], layer_feature, layer_edge_index)

        x_embedding = F.dropout(final_layer, p=self.dp)
        out = self.PredictGCN(x=x_embedding, edge_index=ds_list[-1].edge_index.long().to(self.device))

        return out
