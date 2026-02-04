import torch
import os

# ==============================================================================
# 【修复 PyTorch 2.6+ 加载 OGB 数据集报错的问题】
# ==============================================================================
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load
# ==============================================================================

import argparse
import os.path as osp
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# 引入项目网络定义
import network

# 引入 OGB 和 PyG 相关库
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as TorchDataset

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


# =============================================================================
# 自定义数据集包装器
# =============================================================================
class IndexWrapper(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data.idx = torch.tensor(idx, dtype=torch.long)
        return data


# =============================================================================
# 数据加载部分
# =============================================================================
def data_load(args):
    # 使用 args.dset 动态加载数据集
    dataset = PygGraphPropPredDataset(name=args.dset, root='./data')
    split_idx = dataset.get_idx_split()
    test_idx = split_idx["test"]
    target_dataset_raw = dataset[test_idx]
    target_dataset = IndexWrapper(target_dataset_raw)

    dset_loaders = {}
    dset_loaders["target"] = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.worker)
    dset_loaders["test"] = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.worker)

    print(f"Dataset: {args.dset}")
    print(f"Target size: {len(target_dataset)}")
    return dset_loaders


# =============================================================================
# 【核心修改】计算 ROC-AUC (修复维度错误)
# =============================================================================
def cal_acc(loader, netF, netB, netC, dset_name):
    start_test = True
    evaluator = Evaluator(name=dset_name)
    y_true = []
    y_pred_score = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = netC(netB(netF(data)))  # Output Shape: [Batch, 2]

            # 计算概率分布
            softmax_outputs = nn.Softmax(dim=1)(outputs)

            if start_test:
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

            # 1. 收集真实标签 (保持 [Batch, 1])
            y_true.append(data.y.view(-1, 1).detach().cpu())

            # 2. 收集正类预测概率 (取第2列即 Class 1，并保持 [Batch, 1])
            # softmax_outputs[:, 1] 是正类概率
            y_pred_score.append(softmax_outputs[:, 1].view(-1, 1).detach().cpu())

    # 拼接结果
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred_score, dim=0).numpy()

    # 计算 AUC
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    rocauc = evaluator.eval(input_dict)['rocauc']

    # 计算平均熵
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return rocauc * 100, mean_ent


# =============================================================================
# 核心训练逻辑
# =============================================================================
def train_target(args):
    dset_loaders = data_load(args)

    netF = network.GINVirtual_node(num_layers=5, emb_dim=300).to(device)
    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer,
                                   class_num=2,
                                   bottleneck_dim=args.bottleneck).to(device)

    try:
        modelpath_F = osp.join(args.output_src, 'source_F.pt')
        modelpath_B = osp.join(args.output_src, 'source_B.pt')
        modelpath_C = osp.join(args.output_src, 'source_C.pt')

        netF.load_state_dict(torch.load(modelpath_F, map_location=device))
        netB.load_state_dict(torch.load(modelpath_B, map_location=device))
        netC.load_state_dict(torch.load(modelpath_C, map_location=device))
        print("Loaded pretrained source models successfully.")
    except Exception as e:
        print(f"Warning: Could not load source models ({e}). Using random initialization.")

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    param_group_c = []
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 1}]
    optimizer_c = optim.SGD(param_group_c, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer_c = op_copy(optimizer_c)

    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.bottleneck)
    score_bank = torch.randn(num_sample, 2).to(device)

    netF.eval()
    netB.eval()
    netC.eval()

    print("Initializing banks...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            indx = data.idx

            output = netB(netF(data))
            output_norm = F.normalize(output)
            outputs = netC(output)
            outputs = nn.Softmax(dim=1)(outputs)

            # 注意：如果显存不足，这里可能需要先放到 cpu，使用时再 to(device)
            # 既然 args.dset 数据集都不大，保持在 cpu 或 gpu 均可
            # 为保险起见，这里 fea_bank 保持 CPU，score_bank 保持 GPU (原逻辑)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()

    max_iter = args.max_epoch * len(loader)
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    iter_loader = iter(loader)

    print("Start Training...")
    while iter_num < max_iter:
        try:
            data = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            data = next(iter_loader)

        data = data.to(device)
        if data.num_graphs == 1: continue

        tar_idx = data.idx

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(data))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.K + 1)
            idx_near = idx_near[:, 1:]
            score_near = score_bank[idx_near]

            fea_near = fea_bank[idx_near]
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))
            _, idx_near_near = torch.topk(distance_, dim=-1, largest=True, k=args.KK + 1)
            idx_near_near = idx_near_near[:, :, 1:]

            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1).cpu()
            match = (idx_near_near == tar_idx_).sum(-1).float()
            weight = torch.where(match > 0., match, torch.ones_like(match).fill_(0.1))

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, args.KK)
            weight_kk = weight_kk.fill_(0.1)

            score_near_kk = score_bank[idx_near_near]
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0], -1)
            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1, 2)

        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK, -1)
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk.to(device), reduction='none').sum(-1) *
             weight_kk.to(device)).sum(1))
        loss = torch.mean(const)

        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)
        loss += torch.mean((
                                   F.kl_div(softmax_out_un, score_near.to(device), reduction='none').sum(-1) *
                                   weight.to(device)).sum(1))

        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
        loss += gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()

            # 使用修复后的 cal_acc 和 args.dset
            auc_test, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args.dset)
            log_str = 'Iter:{}/{}; AUC on target = {:.2f}%'.format(iter_num, max_iter, auc_test)

            print(log_str)
            if args.out_file:
                args.out_file.write(log_str + '\n')
                args.out_file.flush()

            netF.train()
            netB.train()
            netC.train()

    return netF, netB, netC


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NRC for Graph SFDA')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id")
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--worker', type=int, default=0, help="CPU workers")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--KK', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    # 新增数据集参数
    parser.add_argument('--dset', type=str, default='ogbg-molhiv', choices=['ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not osp.exists(args.output):
        os.makedirs(args.output)

    args.out_file = open(osp.join(args.output, 'log_target.txt'), 'w')

    train_target(args)