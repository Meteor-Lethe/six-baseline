import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import network  # 引入我们修改后的 network.py

# ==============================================================================
# 【修复 PyTorch 2.6+ 加载 OGB 数据集报错的问题】
# 错误信息: Weights only load failed ... Unsupported global: GLOBAL torch_geometric ...
# 原因: OGB 库调用 torch.load 时未指定 weights_only=False，被新版 PyTorch 拦截。
# 方案: 通过 Monkey Patch 强制将默认行为改回 weights_only=False。
# ==============================================================================
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    # 如果调用方没有显式指定 weights_only，则强制设置为 False 以兼容 OGB
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_load(args):
    # 加载数据集 (使用参数 args.dset)
    dataset = PygGraphPropPredDataset(name=args.dset, root='./data')
    split_idx = dataset.get_idx_split()

    # 源域使用 Train 集
    train_idx = split_idx["train"]
    # 验证使用 Valid 集 (用于监控源域训练过程)
    valid_idx = split_idx["valid"]

    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.worker)
    dset_loaders["source_te"] = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.worker)

    print(f"Dataset: {args.dset}")
    print(f"Source Train size: {len(train_dataset)}, Source Val size: {len(valid_dataset)}")
    return dset_loaders


def cal_acc(loader, netF, netB, netC, dset_name):
    evaluator = Evaluator(name=dset_name)
    y_true = []
    y_pred_score = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = netC(netB(netF(data)))  # Output Shape: [Batch, 2]

            # 计算概率分布 (Softmax)
            softmax_outputs = nn.Softmax(dim=1)(outputs)

            # 收集真实标签 (保持 [Batch, 1])
            y_true.append(data.y.view(-1, 1).detach().cpu())

            # 收集正类预测概率 (取第2列即 Class 1，并保持 [Batch, 1])
            y_pred_score.append(softmax_outputs[:, 1].view(-1, 1).detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred_score, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)['rocauc'] * 100


def train_source(args):
    dset_loaders = data_load(args)

    # 1. 定义网络 (与 train_tar.py 保持一致)
    netF = network.GINVirtual_node(num_layers=5, emb_dim=300).to(device)
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=300, bottleneck_dim=args.bottleneck).to(device)
    # class_num=2 (虽然 OGB 是 1 个输出的 logit，但为了兼容 NRC 架构我们保留 bottleneck->classifier 结构)
    # 注意：这里我们让 classifier 输出 1 个维度，因为是二分类 BCE
    # 或者为了完全兼容 NRC 的 Softmax 逻辑，输出 2 个维度。
    # **关键决策**：为了配合 train_tar.py 中修改后的 Softmax/Cluster 逻辑，我们这里输出 2 维。
    netC = network.feat_classifier(type=args.layer, class_num=2, bottleneck_dim=args.bottleneck).to(device)

    # 2. 优化器 (Graph 任务常用 Adam)
    optimizer = optim.Adam([
        {'params': netF.parameters()},
        {'params': netB.parameters()},
        {'params': netC.parameters()}
    ], lr=args.lr, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()  # 对应输出 2 维

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    # 创建迭代器
    iter_loader = iter(dset_loaders["source_tr"])

    print("Starting Source Training...")
    while iter_num < max_iter:
        try:
            data = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dset_loaders["source_tr"])
            data = next(iter_loader)

        data = data.to(device)
        if data.num_graphs == 1: continue

        iter_num += 1

        # Forward
        features = netB(netF(data))
        outputs = netC(features)  # [batch, 2]

        # Label 处理: OGB 的 data.y 是 [batch, 1] 且是 0/1
        # CrossEntropy 需要 data.y 是 [batch] 且是 long 类型
        target = data.y.view(-1).long()

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            # 使用修复后的 cal_acc 计算验证集 AUC
            acc_val = cal_acc(dset_loaders["source_te"], netF, netB, netC, args.dset)

            print(f"Iter {iter_num}/{max_iter}, Loss: {loss.item():.4f}, Val AUC: {acc_val:.2f}%")

            # 保存模型
            torch.save(netF.state_dict(), osp.join(args.output_dir_src, "source_F.pt"))
            torch.save(netB.state_dict(), osp.join(args.output_dir_src, "source_B.pt"))
            torch.save(netC.state_dict(), osp.join(args.output_dir_src, "source_C.pt"))

            netF.train()
            netB.train()
            netC.train()

    print(f"Source models saved to {args.output_dir_src}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Source Training for Graph')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--max_epoch', type=int, default=20)  # 预训练轮数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output_src', type=str, default='weight/source/')
    # 新增数据集参数
    parser.add_argument('--dset', type=str, default='ogbg-molhiv',
                        choices=['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace'])

    args = parser.parse_args()

    # 简单设置输出路径
    args.output_dir_src = args.output_src
    if not osp.exists(args.output_dir_src):
        os.makedirs(args.output_dir_src)

    train_source(args)