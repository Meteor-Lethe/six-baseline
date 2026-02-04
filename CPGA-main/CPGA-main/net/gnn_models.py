import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# 获取 OGB 定义的原子和边特征维度
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


class AtomEncoder(torch.nn.Module):
    """
    OGB 标准原子编码器 (参考 DoGE，但简化去除了 Soft 逻辑，使用标准 Embedding 以稳定训练)
    """

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding


class BondEncoder(torch.nn.Module):
    """
    OGB 标准边编码器
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        edge_embedding = 0
        for i in range(edge_attr.shape[1]):
            edge_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        return edge_embedding


class GINConv(MessagePassing):
    """
    GIN 卷积层 (参考 DoGE-main)
    """

    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class OGBGNN_Backbone(torch.nn.Module):
    """
    基于 OGB-MolHiv 的 GIN Backbone，带虚拟节点 (Virtual Node)。
    这是特征提取器。
    """

    def __init__(self, num_layers=5, emb_dim=300, dropout=0.5):
        super(OGBGNN_Backbone, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # 虚拟节点 Embedding
        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(emb_dim, 2 * emb_dim),
                    torch.nn.BatchNorm1d(2 * emb_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(2 * emb_dim, emb_dim),
                    torch.nn.BatchNorm1d(emb_dim),
                    torch.nn.ReLU()
                )
            )

        self.pool = global_mean_pool

    def forward(self, batch_data):
        # 解包 PyG 的 Batch 数据对象
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

        # 虚拟节点逻辑
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layers):
            # 增加虚拟节点信息到图节点
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # 消息传递
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_list.append(h)

            # 更新虚拟节点
            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    self.dropout,
                    training=self.training
                )

        node_embedding = h_list[-1]

        # 图级池化 (Graph Pooling) -> 得到图的特征向量
        graph_feature = self.pool(node_embedding, batch)

        return graph_feature


class GNN_Classifier(torch.nn.Module):
    """
    包装类：组合 Backbone 和 Classifier，用于适配 CPGA 的接口
    """

    def __init__(self, backbone, num_classes):
        super(GNN_Classifier, self).__init__()
        self.backbone = backbone
        # 分类头，输入维度必须与 backbone 的 emb_dim 一致
        self.fc = nn.Linear(backbone.emb_dim, num_classes)

    def forward(self, x):
        # x 是 PyG 的 Batch Data 对象
        features = self.backbone(x)  # [batch_size, emb_dim]
        logits = self.fc(features)  # [batch_size, num_classes]
        # CPGA 风格的返回：(logits, features)
        return logits, features