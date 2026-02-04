import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import copy
from tqdm import tqdm

# 引入修改后的 utils
import utils
import loss

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

# ==============================================================================
# 【修复 PyTorch 2.6+ 加载 OGB 数据集报错的问题】
# 保持这个补丁，防止加载 OGB 数据集时报错
# ==============================================================================
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# =============================================================================
#  Data Loading Logic
# =============================================================================

def add_index_to_dataset(dataset):
    """
    【关键修复】：
    PyG 的 Dataset 在切片或索引时（如 dataset[i]）往往会返回一个新的 Data 对象副本/视图。
    直接在 dataset 循环中修改属性无法持久化。

    解决方法：将 Dataset 显式转换为一个 Data 对象的列表 (List)，
    并在列表中为每个对象永久添加 'idx' 属性。
    PyG 的 DataLoader 支持直接读取 Data List。
    """
    data_list = []
    print(f"Processing target dataset to add indices...")
    for i in range(len(dataset)):
        # 获取第 i 个图数据对象
        # 显式调用 .clone() 确保我们拥有一个独立的可修改对象（虽然通常 dataset[i] 已经是新的）
        data = dataset[i].clone()

        # 添加全局索引 'idx'
        # 必须转为 Tensor，这样 PyG 的 DataLoader 才会自动将其拼接成 Batch 中的张量
        data.idx = torch.tensor(i, dtype=torch.long)

        data_list.append(data)

    return data_list


def data_load(args):
    print(f"Loading {args.dset} dataset...")
    # 加载 OGB 数据集
    dataset = PygGraphPropPredDataset(name=args.dset, root='./data')

    # 获取 OGB 官方定义的 Scaffold Split
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    valid_idx = split_idx["valid"]
    test_idx = split_idx["test"]

    # 定义 Source (源域) 和 Target (目标域)
    # Source: Train Set
    source_dataset = dataset[train_idx]

    # Target: Test Set (模拟无标签目标域)
    # 注意：这里我们先获取子集，然后立即转换为带有 idx 的列表
    target_subset = dataset[test_idx]
    target_dataset = add_index_to_dataset(target_subset)

    # Test: Test Set (用于验证，不需要 idx)
    test_dataset = dataset[test_idx]

    print(f"Source samples: {len(source_dataset)}")
    print(f"Target samples: {len(target_dataset)}")

    dset_loaders = {}
    # DataLoader
    dset_loaders["source"] = DataLoader(source_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.worker, drop_last=True)

    # Target DataLoader 读取的是 List[Data]，行为与 Dataset 一致
    dset_loaders["target"] = DataLoader(target_dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.worker, drop_last=True)

    dset_loaders["test"] = DataLoader(test_dataset, batch_size=args.batch_size * 2,
                                      shuffle=False, num_workers=args.worker)

    return dset_loaders


# =============================================================================
#  Scheduler & Loss Utils
# =============================================================================

def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    """
    学习率衰减策略
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


# =============================================================================
#  Training Logic
# =============================================================================

def train(args):
    # 修改为使用 GPU
    device = torch.device('cuda')
    print(f"Running on device: {device}")

    # 加载数据
    dset_loaders = data_load(args)

    # OGB-Molhiv 是二分类任务
    args.class_num = 2

    # 1. 初始化网络
    # netG: 图特征提取器
    netG = utils.GNNBase(emb_dim=args.emb_dim, num_layers=5).to(device)

    # netF: 分类器头
    netF = utils.ResClassifier(class_num=args.class_num, feature_dim=args.emb_dim,
                               bottleneck_dim=args.bottleneck_dim).to(device)

    # 计算最大迭代次数
    max_len = max(len(dset_loaders["source"]), len(dset_loaders["target"]))
    args.max_iter = args.max_epoch * max_len
    print(f"Max Iterations: {args.max_iter}")

    # 对抗网络初始化
    ad_flag = False
    if args.method in {'DANN', 'DANNE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim, 1024, max_iter=args.max_iter).to(device)
        ad_flag = True
    if args.method in {'CDAN', 'CDANE'}:
        ad_net = utils.AdversarialNetwork(args.bottleneck_dim * args.class_num, 1024, max_iter=args.max_iter).to(device)
        random_layer = None
        ad_flag = True

    # 优化器
    optimizer_g = optim.SGD(netG.parameters(), lr=args.lr * 0.1)
    optimizer_f = optim.SGD(netF.parameters(), lr=args.lr)
    if ad_flag:
        optimizer_d = optim.SGD(ad_net.parameters(), lr=args.lr)

    # Memory Bank 初始化
    num_target_samples = len(dset_loaders["target"].dataset)
    print(f"Initializing Memory Bank for {num_target_samples} target samples...")

    if args.pl.startswith('atdoc_na'):
        mem_fea = torch.rand(num_target_samples, args.bottleneck_dim).to(device)
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)
        mem_cls = torch.ones(num_target_samples, args.class_num).to(device) / args.class_num

    if args.pl == 'atdoc_nc':
        mem_fea = torch.rand(args.class_num, args.bottleneck_dim).to(device)
        mem_fea = mem_fea / torch.norm(mem_fea, p=2, dim=1, keepdim=True)

    # 迭代器初始化
    source_loader_iter = iter(dset_loaders["source"])
    target_loader_iter = iter(dset_loaders["target"])

    list_acc = []

    # ------------------ 训练循环 ------------------
    for iter_num in range(1, args.max_iter + 1):
        netG.train()
        netF.train()

        # 调整学习率
        lr_scheduler(optimizer_g, init_lr=args.lr * 0.1, iter_num=iter_num, max_iter=args.max_iter)
        lr_scheduler(optimizer_f, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)
        if ad_flag:
            lr_scheduler(optimizer_d, init_lr=args.lr, iter_num=iter_num, max_iter=args.max_iter)

        # 获取 Source Batch
        try:
            batch_source = next(source_loader_iter)
        except StopIteration:
            source_loader_iter = iter(dset_loaders["source"])
            batch_source = next(source_loader_iter)

        # 获取 Target Batch
        try:
            batch_target = next(target_loader_iter)
        except StopIteration:
            target_loader_iter = iter(dset_loaders["target"])
            batch_target = next(target_loader_iter)

        # 数据移至 GPU
        batch_source = batch_source.to(device)
        batch_target = batch_target.to(device)

        # 获取 Target 的索引 (idx)
        # 现在的 batch_target 是由我们手动添加了 idx 的 Data 对象拼接而成的
        # 因此它一定包含 idx 属性
        idx = batch_target.idx

        # Forward Pass
        # 1. Source
        feat_src_graph = netG(batch_source.x, batch_source.edge_index, batch_source.edge_attr, batch_source.batch)
        feat_src, out_src = netF(feat_src_graph)

        # 2. Target
        feat_tgt_graph = netG(batch_target.x, batch_target.edge_index, batch_target.edge_attr, batch_target.batch)
        feat_tgt, out_tgt = netF(feat_tgt_graph)

        # 拼接用于 Loss 计算
        features = torch.cat((feat_src, feat_tgt), dim=0)
        outputs = torch.cat((out_src, out_tgt), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        # A. Transfer Loss
        eff = utils.calc_coeff(iter_num, max_iter=args.max_iter)

        if args.method == 'srconly':
            transfer_loss = torch.tensor(0.0).to(device)
        elif args.method in {'DANN', 'DANNE'}:
            entropy = loss.Entropy(softmax_out) if args.method == 'DANNE' else None
            transfer_loss = loss.DANN(features, ad_net, entropy, eff)
        elif args.method in {'CDAN', 'CDANE'}:
            entropy = loss.Entropy(softmax_out) if args.method == 'CDANE' else None
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, eff, random_layer)
        else:
            transfer_loss = torch.tensor(0.0).to(device)

        # B. Classifier Loss
        labels_source = batch_source.y.squeeze()
        if labels_source.dtype != torch.long:
            labels_source = labels_source.long()

        # 处理可能的 Shape 问题：如果是空 batch 或 1 个样本
        if labels_source.dim() == 0:
            labels_source = labels_source.unsqueeze(0)

        classifier_loss = nn.CrossEntropyLoss()(out_src, labels_source)

        total_loss = transfer_loss + classifier_loss

        # C. ATDOC Loss
        eff_pl = iter_num / args.max_iter

        if args.pl == 'none':
            pass
        elif args.pl.startswith('atdoc_na'):
            # 计算距离
            dis = -torch.mm(feat_tgt.detach(), mem_fea.t())  # [B, N_t]

            # 排除自身
            for di in range(dis.size(0)):
                if idx[di] < mem_fea.size(0):
                    dis[di, idx[di]] = -1e10

                    # Top-K
            _, p1 = torch.sort(dis, dim=1, descending=True)

            # 加权
            w = torch.zeros(feat_tgt.size(0), mem_fea.size(0)).to(device)
            for wi in range(w.size(0)):
                for wj in range(args.K):
                    neighbor_idx = p1[wi, wj]
                    w[wi][neighbor_idx] = 1 / args.K

            # 伪标签
            weight_, pred = torch.max(w.mm(mem_cls), 1)

            loss_pl = nn.CrossEntropyLoss(reduction='none')(out_tgt, pred)
            classifier_loss_pl = torch.sum(weight_ * loss_pl) / (torch.sum(weight_).item() + 1e-8)

            total_loss += args.tar_par * eff_pl * classifier_loss_pl

        # Backprop
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if ad_flag:
            optimizer_d.zero_grad()

        total_loss.backward()

        optimizer_g.step()
        optimizer_f.step()
        if ad_flag:
            optimizer_d.step()

        # Update Memory Bank
        with torch.no_grad():
            if args.pl.startswith('atdoc_na'):
                netG.eval()
                netF.eval()
                f_t_g = netG(batch_target.x, batch_target.edge_index, batch_target.edge_attr, batch_target.batch)
                f_t, o_t = netF(f_t_g)

                f_t = f_t / torch.norm(f_t, p=2, dim=1, keepdim=True)
                sm_out = nn.Softmax(dim=1)(o_t)

                # Update
                mem_fea[idx] = (1.0 - args.momentum) * mem_fea[idx] + args.momentum * f_t.clone()
                mem_cls[idx] = (1.0 - args.momentum) * mem_cls[idx] + args.momentum * sm_out.clone()

                netG.train()
                netF.train()

        # Logging
        if iter_num % 100 == 0:
            log_str = f'Iter:{iter_num}/{args.max_iter}; Total Loss: {total_loss.item():.4f}; Cls Loss: {classifier_loss.item():.4f}'
            print(log_str)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()

        # Validation
        if iter_num % int(args.max_iter / 10) == 0 and iter_num > 0:
            netG.eval()
            netF.eval()
            acc, _, _, _ = utils.cal_acc(dset_loaders["test"], netG, netF, device)
            #log_str = f'Task: {args.dset}, Iter:{iter_num}; Accuracy = {acc * 100:.2f}%'
            log_str = f'Task: {args.dset}, Iter:{iter_num}; ROC-AUC = {acc * 100:.2f}%'
            print(log_str)
            args.out_file.write(log_str + '\n')
            list_acc.append(acc * 100)

    # End
    final_acc = list_acc[-1] if len(list_acc) > 0 else 0.0
    print(f"\n==========================================")
    print(f"Final ROC-AUC: {final_acc:.2f}%")
    print(f"==========================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ATDOC for OGBG-Molhiv (PyTorch GPU Version)')
    parser.add_argument('--method', type=str, default='srconly',
                        choices=['srconly', 'CDAN', 'CDANE', 'DANN', 'DANNE'])
    parser.add_argument('--pl', type=str, default='atdoc_na',
                        choices=['none', 'atdoc_na', 'atdoc_nc'])
    parser.add_argument('--dset', type=str, default='ogbg-molhiv', help="Dataset name")
    parser.add_argument('--output', type=str, default='logs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--tar_par', type=float, default=1.0)
    # 新增 gpu_id 参数以配合 CUDA 适配
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")

    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    torch.manual_seed(args.seed)
    # CUDA 随机种子
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.output_dir = osp.join(args.output, args.dset, args.method, args.pl)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, "log.txt"), "w")
    utils.print_args(args)

    train(args)