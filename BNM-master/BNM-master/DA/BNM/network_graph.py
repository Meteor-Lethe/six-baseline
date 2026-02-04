import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    """
    Project 1 中的 GINConv 复用
    """

    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        # GIN update: MLP((1+eps)*x + aggregate(x_j + e_ij))
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GINVirtual_node(torch.nn.Module):
    """
    Project 1 中的 GINVirtual_node 复用
    用于生成节点嵌入 (Node Embeddings)
    """

    def __init__(self, num_layers, emb_dim, dropout=0.5):
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # 使用标准的 AtomEncoder 处理 ogb 的离散特征
        self.atom_encoder = AtomEncoder(emb_dim)

        # Virtual node embedding
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
                torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                    torch.nn.BatchNorm1d(2 * emb_dim),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(2 * emb_dim, emb_dim),
                                    torch.nn.BatchNorm1d(emb_dim),
                                    torch.nn.ReLU())
            )

    def forward(self, x, edge_index, edge_attr, batch):
        # Virtual node input
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layers):
            # Add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # Message passing
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h_list.append(h)

            # Update virtual nodes
            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    self.dropout,
                    training=self.training
                )

        return h_list[-1]


class GIN_BNM(torch.nn.Module):
    """
    修改后的主模型，专为 BNM 设计。
    结构：GIN + Virtual Node + Classifier
    输出：(features, logits)
    """

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, dropout=0.5):
        super(GIN_BNM, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        # 1. GNN Backbone (from Project 1)
        self.gnn_node = GINVirtual_node(num_layers, emb_dim, dropout=dropout)

        # 2. Pooling
        self.pool = global_mean_pool

        # 3. Classifier (Linear)
        # 注意：Project 1 是二分类 (molhiv)，通常输出维度是 1 (BCEWithLogits) 或 2 (CrossEntropy)
        # BNM 原代码使用的是 CrossEntropyLoss，且 label 是 long 类型。
        # molhiv 的 label 是 0/1。为了适配 BNM 的 softmax 逻辑，这里我们将输出维度设为 2。
        self.classifier = torch.nn.Linear(self.emb_dim, 2)

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch

        # 提取节点特征
        h_node = self.gnn_node(x, edge_index, edge_attr, batch)

        # 图级池化 -> Feature (用于 BNM 可以在特征层做对齐，或者仅使用 Logits)
        h_graph = self.pool(h_node, batch)

        # 分类器 -> Logits
        logits = self.classifier(h_graph)

        # 返回 (Features, Logits) 以适配 BNM 训练循环
        return h_graph, logits

    def get_parameters(self):
        """
        适配 BNM 的优化器参数获取接口
        """
        parameter_list = [
            {"params": self.gnn_node.parameters(), "lr_mult": 1, 'decay_mult': 2},
            {"params": self.classifier.parameters(), "lr_mult": 10, 'decay_mult': 2}
        ]
        return parameter_list