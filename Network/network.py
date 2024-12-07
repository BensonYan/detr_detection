import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GraphNorm

class MultiHeadGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiHeadGCNLayer, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([GCNConv(in_channels, out_channels) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * out_channels, out_channels)

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=1)  # Concatenate all head outputs
        return self.linear(concatenated)  # Project concatenated outputs to output dimension

class MultiHeadGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_heads=4, num_layers=3):
        super(MultiHeadGCN, self).__init__()
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(MultiHeadGCNLayer(in_channels, hidden_channels, num_heads))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(MultiHeadGCNLayer(hidden_channels, hidden_channels, num_heads))
        # Output layer
        self.layers.append(MultiHeadGCNLayer(hidden_channels, out_channels, num_heads))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x, edge_index, batch = data[0], data[1], data[2]
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        # x = self.layers[-1](x, edge_index)  # No activation on the output layer
        x = global_mean_pool(x, batch)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNClassifier, self).__init__()
        # 定义 GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # 定义 GraphNorm
        self.graph_norm1 = GraphNorm(hidden_dim)
        # self.graph_norm2 = GraphNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 16)
        # self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x, batch).relu()
        # x = self.conv2(x, edge_index)
        # x = self.graph_norm2(x, batch).relu()
        # 使用 global_mean_pool，根据 batch 进行池化
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out



    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.graph_norm1(x, batch).relu()
        # x = self.conv2(x, edge_index)
        # x = self.graph_norm2(x, batch).relu()
        # 使用 global_mean_pool，根据 batch 进行池化
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x