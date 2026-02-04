import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import random
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from tqdm import tqdm

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

# 设置设备 (适配 GPU 环境)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = (1 + args.lr_gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def data_load(args):
    print(f"Loading OGB Dataset: {args.dset}...")
    dataset = PygGraphPropPredDataset(name=args.dset, root='./data')
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.worker)
    dset_loaders["source_te"] = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.worker)
    return dset_loaders


def cal_rocauc(loader, netF, netB, netC, dset_name):
    """使用 OGB 官方 Evaluator 计算 ROC-AUC"""
    evaluator = Evaluator(name=dset_name)
    all_output_list = []
    all_label_list = []

    netF.eval()
    netB.eval()
    netC.eval()

    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.to(device)
            outputs = netC(netB(netF(data)))

            # 记录正类概率 (Index 1)
            probs = torch.softmax(outputs, dim=1)[:, 1].view(-1, 1)

            all_output_list.append(probs.cpu())
            all_label_list.append(data.y.cpu())

    y_true = torch.cat(all_label_list, dim=0)
    y_pred = torch.cat(all_output_list, dim=0)

    # 【修复点】：键名必须为 "y_pred" 以适配 OGB Evaluator
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict['rocauc']


def train_source(args):
    netF = network.GINBase(emb_dim=300, num_layers=5).to(device)
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device)

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 1.0}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate * 1.0}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    criterion = nn.CrossEntropyLoss()
    dset_loaders = data_load(args)

    netF.train()
    netB.train()
    netC.train()

    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    print("Source Training Started...")
    iter_source = iter(dset_loaders["source_tr"])

    while iter_num < max_iter:
        try:
            data_batch = next(iter_source)
        except StopIteration:
            iter_source = iter(dset_loaders["source_tr"])
            data_batch = next(iter_source)

        if data_batch.num_graphs == 1: continue

        data_batch = data_batch.to(device)
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs = netC(netB(netF(data_batch)))
        labels = data_batch.y.to(torch.long).flatten()

        is_labeled = (data_batch.y == data_batch.y).flatten()
        loss = criterion(outputs[is_labeled], labels[is_labeled])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print(f'Iter {iter_num}/{max_iter}, Loss: {loss.item():.4f}')
            roc_auc = cal_rocauc(dset_loaders["source_te"], netF, netB, netC, args.dset)
            print(f'Validation ROC-AUC: {roc_auc:.4f}')
            netF.train()
            netB.train()
            netC.train()

    torch.save(netF.state_dict(), osp.join(args.output_dir_src, 'source_F.pt'))
    torch.save(netB.state_dict(), osp.join(args.output_dir_src, 'source_B.pt'))
    torch.save(netC.state_dict(), osp.join(args.output_dir_src, 'source_C.pt'))
    print("Source models saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Source Training for OGB')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--dset', type=str, default='ogbg-molhiv')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--output_src', type=str, default='pretrained_models')
    parser.add_argument('--lr_gamma', type=float, default=10.0)
    parser.add_argument('--lr_power', type=float, default=0.75)

    args = parser.parse_args()

    if args.dset in ['ogbg-molhiv', 'ogbg-bbbp', 'ogbg-bace']:
        args.class_num = 2
    else:
        args.class_num = 2

    args.output_dir_src = osp.join(args.output_src, args.dset)
    if not osp.exists(args.output_dir_src):
        os.makedirs(args.output_dir_src, exist_ok=True)

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_source(args)