import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


# ==============================================================================
# [PART 1] SHOT Original Utilities & Classes
# ==============================================================================

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # 【修复】Numpy 2.0 移除了 np.float，改为使用 python 原生 float
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn,
            "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d,
            "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


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


class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x


class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y


# ==============================================================================
# [PART 2] OGB/DoGE GNN Extensions
# ==============================================================================

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


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
        self.emb_dim = emb_dim
        self.weights = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(dim, emb_dim)) for dim in full_atom_feature_dims
        ])
        for w in self.weights:
            torch.nn.init.xavier_uniform_(w)

    def forward(self, x_soft_list):
        N = x_soft_list[0].size(0)
        emb = torch.zeros((N, self.emb_dim), device=self.weights[0].device)
        for i, w in enumerate(self.weights):
            emb = emb + x_soft_list[i] @ w
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
        emb = torch.zeros((E, self.emb_dim), device=device)
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
    def __init__(self, num_layers=5, emb_dim=300, dropout=0.5, encode_node=True):
        super(GINVirtual_node, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.encode_node = encode_node
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

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # 初始化虚拟节点 embedding 变量
        vn_embedding_tensor = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        if self.encode_node:
            x_soft_list = int_to_soft_list_from_continuous_x(x, full_atom_feature_dims, device=x.device)
            h_list = [self.atom_encoder(x_soft_list)]
        else:
            h_list = [x]

        for layer in range(self.num_layers):
            # 将虚拟节点信息加到当前层节点特征上
            h_list[layer] = h_list[layer] + vn_embedding_tensor[batch]

            # GIN 卷积
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            # 激活与Dropout
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            # 【关键】必须确保 append 每次都执行，不要缩进到 if/else 里
            h_list.append(h)

            # 更新虚拟节点特征
            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + vn_embedding_tensor
                vn_embedding_tensor = F.dropout(
                    self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                    self.dropout, training=self.training
                )

        h_node = h_list[-1]
        h_graph = global_mean_pool(h_node, batch)
        return h_graph


class GraphShotNet(nn.Module):
    def __init__(self, feature_dim=300, bottleneck_dim=256, class_num=2, dropout=0.5):
        super(GraphShotNet, self).__init__()
        self.base_network = GINVirtual_node(num_layers=5, emb_dim=feature_dim, dropout=dropout)
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.bottleneck[0].weight.data.normal_(0, 0.005)
        self.bottleneck[0].bias.data.fill_(0.1)
        self.fc = nn.utils.weight_norm(nn.Linear(bottleneck_dim, class_num), dim=None)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

    def forward(self, inputs):
        features = self.base_network(inputs)
        features = self.bottleneck(features)
        outputs = self.fc(features)
        return outputs, features