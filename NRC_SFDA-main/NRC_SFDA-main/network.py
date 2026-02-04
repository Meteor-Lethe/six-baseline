import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.weight_norm as weightNorm
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# =============================================================================
# 第一部分：来自 DoGE 的 GNN 组件 (复用 AtomEncoderSoft, BondEncoderSoft, GIN)
# =============================================================================

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


def int_to_soft_list_from_continuous_x(x, full_dims, device, tau=1.0):
    onehot_soft_list = []
    for i, dim in enumerate(full_dims):
        xi = x[:, i:i + 1].float()
        positions = torch.arange(dim, dtype=torch.float32, device=device).view(1, -1)
        logits = (xi / (dim - 1.0)) * positions
        soft = F.softmax(logits / tau, dim=1)
        onehot_soft_list.append(soft)
    return onehot_soft_list


class AtomEncoderSoft(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoderSoft, self).__init__()
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim, emb_dim)) for dim in full_atom_feature_dims
        ])
        for w in self.weights: torch.nn.init.xavier_uniform_(w)

    def forward(self, x_soft_list):
        N = x_soft_list[0].size(0)
        emb = torch.zeros((N, self.weights[0].size(1)), device=x_soft_list[0].device)
        for i, w in enumerate(self.weights):
            emb = emb + x_soft_list[i] @ w
        return emb


class BondEncoderSoft(torch.nn.Module):
    def __init__(self, emb_dim, tau=1.0):
        super(BondEncoderSoft, self).__init__()
        self.full_dims = full_bond_feature_dims
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim, emb_dim)) for dim in self.full_dims
        ])
        self.tau = tau
        for w in self.weights: torch.nn.init.xavier_uniform_(w)

    def forward(self, edge_attr):
        device = edge_attr.device
        emb = torch.zeros((edge_attr.size(0), self.weights[0].size(1)), device=device)
        for i, dim in enumerate(self.full_dims):
            xi = edge_attr[:, i:i + 1].float()
            positions = torch.arange(dim, dtype=torch.float32, device=device).view(1, -1)
            logits = (xi / (dim - 1.0)) * positions
            soft = F.softmax(logits / self.tau, dim=1)
            emb = emb + soft @ self.weights[i]
        return emb


class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoderSoft(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


# 这是 GNN 的 Backbone，对应 NRC 中的 ResBase
class GINVirtual_node(torch.nn.Module):
    def __init__(self, num_layers=5, emb_dim=300, dropout=0.5):
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim

        if self.num_layers < 2: raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoderSoft(emb_dim)
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim), torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU()
            ))

        # 兼容 NRC：添加一个 output dimension 属性，虽然 GNN 最终输出由 emb_dim 决定
        self.in_features = emb_dim

    def forward(self, data):
        # 适配 PyG 的 Batch 数据结构
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 虚拟节点处理
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # 节点编码
        x_soft_list = int_to_soft_list_from_continuous_x(x, full_atom_feature_dims, device=x.device)
        h_list = [self.atom_encoder(x_soft_list)]

        for layer in range(self.num_layers):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)

            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_mean_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                  self.dropout, training=self.training)

        node_embedding = h_list[-1]
        # 全局池化得到图表示
        graph_embedding = global_mean_pool(node_embedding, batch)
        return graph_embedding


# =============================================================================
# 第二部分：NRC 原始组件 (保留 Bottleneck 和 Classifier)
# =============================================================================

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x