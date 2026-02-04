import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import numpy as np
from torch_geometric.nn import global_mean_pool, MessagePassing, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


# ============================================================================
# Part 1: GNN Components (移植自 Project 1 DoGE，用于处理 OGBG 数据)
# ============================================================================

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINVirtual_node(torch.nn.Module):
    def __init__(self, num_layers, emb_dim, dropout=0.5, encode_node=True):
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node

        self.atom_encoder = AtomEncoder(emb_dim)
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

    def forward(self, x, edge_index, edge_attr, batch):
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        if self.encode_node:
            h_list = [self.atom_encoder(x)]
        else:
            h_list = [x]

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
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    self.dropout, training=self.training
                )

        node_embedding = h_list[-1]
        return node_embedding


# ============================================================================
# Part 2: Feature Extractor Wrapper (替代原有的 ResBase/VGGBase)
# ============================================================================

class GINBase(nn.Module):
    """
    OGBG 特征提取器 backbone。
    输入: PyG Batch Data 对象
    输出: Graph Embedding (Tensor [Batch_Size, Emb_Dim])
    """

    def __init__(self, emb_dim=300, num_layers=5, dropout=0.5):
        super(GINBase, self).__init__()
        # 使用 Project 1 的 GIN 结构
        self.gnn_node = GINVirtual_node(num_layers=num_layers, emb_dim=emb_dim, dropout=dropout)
        self.pool = global_mean_pool
        self.in_features = emb_dim  # 供后续 Bottleneck 层读取维度

    def forward(self, data):
        # 提取节点特征
        h_node = self.gnn_node(data.x, data.edge_index, data.edge_attr, data.batch)
        # 池化得到图特征
        h_graph = self.pool(h_node, data.batch)
        return h_graph


# ============================================================================
# Part 3: Utils & Classifier Heads (保留 CoWA 原有逻辑)
# ============================================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
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

# 下面保留原有的ResNet/VGG定义以防代码依赖报错，但在OGB任务中不会被调用
# ... (User provided ResBase/VGGBase code can remain here or be removed)