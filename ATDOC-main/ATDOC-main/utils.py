import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# 获取 OGB 的特征维度常量
full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


# =============================================================================
#  来自 DoGE 的 GNN 组件 (AtomEncoderSoft, BondEncoderSoft, GINConv)
# =============================================================================

def int_to_soft_list_from_continuous_x(x, full_dims, device, tau=1.0):
    """辅助函数：将离散特征转换为软编码列表（保持接口兼容）"""
    onehot_soft_list = []
    N = x.size(0)
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
        self.full_dims = full_atom_feature_dims
        self.emb_dim = emb_dim
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim, emb_dim)) for dim in full_atom_feature_dims
        ])
        for w in self.weights:
            torch.nn.init.xavier_uniform_(w)

    def forward(self, x_soft_list):
        N = x_soft_list[0].size(0)
        emb = torch.zeros((N, self.emb_dim), device=self.weights[0].device, dtype=self.weights[0].dtype)
        for i, w in enumerate(self.weights):
            xi = x_soft_list[i]
            emb = emb + xi @ w
        return emb


class BondEncoderSoft(torch.nn.Module):
    def __init__(self, emb_dim, tau=1.0):
        super(BondEncoderSoft, self).__init__()
        self.full_dims = full_bond_feature_dims
        self.emb_dim = emb_dim
        self.tau = tau
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim, emb_dim)) for dim in self.full_dims
        ])
        for w in self.weights:
            torch.nn.init.xavier_uniform_(w)

    def forward(self, edge_attr):
        E = edge_attr.size(0)
        device = edge_attr.device
        emb = torch.zeros((E, self.emb_dim), device=device, dtype=self.weights[0].dtype)
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
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
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


class GINVirtual_node(torch.nn.Module):
    """
    DoGE 中的 GIN 实现，包含 Virtual Node。
    作为 ATDOC 的 Backbone (Generator)。
    """

    def __init__(self, num_layers=5, emb_dim=300, dropout=0.5, encode_node=True):
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
        self.emb_dim = emb_dim

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

    def forward(self, x, edge_index, edge_attr, batch):
        # Virtual Node Init
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        # Node Encoding
        if self.encode_node:
            x_soft_list = int_to_soft_list_from_continuous_x(x, full_atom_feature_dims, device=x.device)
            h_list = [self.atom_encoder(x_soft_list)]
        else:
            h_list = [x]

        # Message Passing
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

        return h_list[-1]


# =============================================================================
#  ATDOC 适配封装类
# =============================================================================

class GNNBase(nn.Module):
    """
    封装 GIN，适配 ATDOC 的 netG 接口。
    输出图级别的 Embedding (Global Pooling 后)。
    """

    def __init__(self, emb_dim=300, num_layers=5, dropout=0.5):
        super(GNNBase, self).__init__()
        self.gnn_node = GINVirtual_node(num_layers=num_layers, emb_dim=emb_dim, dropout=dropout)
        self.pool = global_mean_pool
        self.in_features = emb_dim  # 暴露给 netF 使用

    def forward(self, x, edge_index, edge_attr, batch):
        h_node = self.gnn_node(x, edge_index, edge_attr, batch)
        h_graph = self.pool(h_node, batch)
        return h_graph


class ResClassifier(nn.Module):
    """
    分类器头：Feature (300d) -> Bottleneck (256d) -> Class Logits
    """

    def __init__(self, class_num, feature_dim, bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        features = self.bottleneck(x)
        outputs = self.fc(features)
        return features, outputs


class AdversarialNetwork(nn.Module):
    """
    用于 DANN/CDAN 的对抗网络
    """

    def __init__(self, in_feature, hidden_size, max_iter=10000.0):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.max_iter, self.high, self.low, self.alpha)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y


def calc_coeff(iter_num, max_iter, high=1.0, low=0.0, alpha=10.0):
    return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


# 评估函数辅助 (Acc)
# def cal_acc(loader, netG, netF, device):
#     start_test = True
#     with torch.no_grad():
#         iter_test = iter(loader)
#         for i in range(len(loader)):
#             data = next(iter_test)
#             data = data.to(device)
#             # GNN Forward
#             features = netG(data.x, data.edge_index, data.edge_attr, data.batch)
#             # Classifier Forward
#             _, outputs = netF(features)
#
#             if start_test:
#                 all_output = outputs.float().cpu()
#                 all_label = data.y.float().cpu()
#                 start_test = False
#             else:
#                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
#                 all_label = torch.cat((all_label, data.y.float().cpu()), 0)
#
#     _, predict = torch.max(all_output, 1)
#     accuracy = torch.sum(torch.squeeze(predict).float() == all_label.squeeze()).item() / float(all_label.size(0))
#
#     return accuracy, predict, all_output, all_label

def cal_acc(loader, netG, netF, device):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            data = data.to(device)
            # GNN Forward
            features = netG(data.x, data.edge_index, data.edge_attr, data.batch)
            # Classifier Forward
            _, outputs = netF(features)

            if start_test:
                all_output = outputs.float().cpu()
                all_label = data.y.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, data.y.float().cpu()), 0)

    # 原来的 Accuracy 计算逻辑（保留备用，或者直接注释掉）
    _, predict = torch.max(all_output, 1)
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label.squeeze()).item() / float(all_label.size(0))

    # 新增 ROC-AUC 计算逻辑
    # 1. 对 logits 进行 softmax 获取概率
    probs = F.softmax(all_output, dim=1)

    # 2. 取出“阳性类 (Class 1)”的概率列
    pos_probs = probs[:, 1]

    # 3. 计算 AUC (需要确保 tensor 转为 numpy)
    # 注意：all_label 可能需要 squeeze 处理维度匹配问题
    try:
        auc = roc_auc_score(all_label.squeeze().numpy(), pos_probs.numpy())
    except ValueError:
        # 这种异常通常发生在测试集里只有一个类别（比如全是负样本）导致无法计算 AUC
        auc = 0.0

    # 返回 auc 代替原来的 accuracy
    return auc, predict, all_output, all_label