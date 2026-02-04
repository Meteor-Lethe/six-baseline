import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import random
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import matplotlib
import math

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

matplotlib.use('Agg')
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
    dataset = PygGraphPropPredDataset(name=args.dset, root='./data')
    split_idx = dataset.get_idx_split()
    raw_target_dataset = dataset[split_idx["test"]]

    target_data_list = []
    for i in range(len(raw_target_dataset)):
        data = raw_target_dataset[i].clone()
        data.local_sample_idx = torch.tensor([i], dtype=torch.long)
        target_data_list.append(data)

    dset_loaders = {}
    dset_loaders["target"] = DataLoader(target_data_list, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.worker)
    dset_loaders["test"] = DataLoader(target_data_list, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.worker)
    return dset_loaders


def gmm(all_fea, pi, mu, all_output, args):
    log_probs = []
    epsilon_eye = args.epsilon * torch.eye(all_fea.shape[1]).to(all_fea.device)
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:, i].unsqueeze(dim=-1)
        denominator = predi.sum() + 1e-8
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / denominator + epsilon_eye
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += epsilon_eye * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        logdet = torch.logdet(Covi)
        temp_transformed = torch.matmul(temp, chol_inv.t())
        mah_dist = (temp_transformed ** 2).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        log_probs.append(log_prob)
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    return zz, torch.exp(zz)


def evaluation(loader, netF, netB, netC, args):
    netF.eval()
    netB.eval()
    netC.eval()
    all_fea_list, all_output_list, all_label_list = [], [], []
    evaluator = Evaluator(name=args.dset)

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluation"):
            data = data.to(device)
            feas = netB(netF(data))
            outputs = netC(feas)
            all_fea_list.append(feas.float().cpu())
            all_output_list.append(outputs.float().cpu())
            all_label_list.append(data.y.float().cpu())

    all_fea = torch.cat(all_fea_list, dim=0).to(device)
    all_output = torch.cat(all_output_list, dim=0).to(device)
    all_label = torch.cat(all_label_list, dim=0).to(device)
    all_output_softmax = nn.Softmax(dim=1)(all_output)

    # 【修复点】：键名修改为 "y_pred"
    y_true = all_label.view(-1, 1).long()
    y_pred = all_output_softmax[:, 1].view(-1, 1)
    roc_auc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})['rocauc']

    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    pi = all_output_softmax.sum(dim=0)
    mu = torch.matmul(all_output_softmax.t(), all_fea) / (pi.unsqueeze(dim=-1) + 1e-8)
    zz, gamma = gmm(all_fea, pi, mu, torch.ones(len(all_fea), args.class_num).to(device) / args.class_num, args)

    for _ in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), all_fea) / (pi.unsqueeze(dim=-1) + 1e-8)
        zz, gamma = gmm(all_fea, pi, mu, gamma, args)

    log_str = 'Model Prediction : ROC-AUC = {:.4f}'.format(roc_auc)
    if args.out_file:
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
    print(log_str)

    sort_zz = zz.sort(dim=1, descending=True)[0]
    LPG = (sort_zz[:, 0] - sort_zz[:, 1]) / ((sort_zz[:, 0] - sort_zz[:, 1]).max() + 1e-8)
    return gamma, LPG, roc_auc


def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    return ((- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1) * coeff).mean(dim=0)


def feature_mixup(features, c_batch, t_batch, netB, netC, args):
    if args.alpha == 0:
        return KLLoss(netC(netB(features)), t_batch, c_batch, args)
    lam = torch.from_numpy(np.random.beta(args.alpha, args.alpha, [len(features)])).float().to(device).view(-1, 1)
    shuffle_idx = torch.randperm(len(features))
    mixed_feat = lam * features + (1 - lam) * features[shuffle_idx]
    mixed_c = lam.squeeze() * c_batch + (1 - lam.squeeze()) * c_batch[shuffle_idx]
    mixed_t = lam * t_batch + (1 - lam) * t_batch[shuffle_idx]
    return KLLoss(netC(netB(mixed_feat)), mixed_t, mixed_c, args)


def train_target(args):
    netF = network.GINBase(emb_dim=300, num_layers=5).to(device)
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device)

    netF.load_state_dict(torch.load(osp.join(args.output_dir_src, 'source_F.pt'), map_location=device))
    netB.load_state_dict(torch.load(osp.join(args.output_dir_src, 'source_B.pt'), map_location=device))
    netC.load_state_dict(torch.load(osp.join(args.output_dir_src, 'source_C.pt'), map_location=device))

    optimizer = optim.SGD([
        {'params': netF.parameters(), 'lr': args.lr * args.lr_decay1},
        {'params': netB.parameters(), 'lr': args.lr * args.lr_decay2},
        {'params': netC.parameters(), 'lr': args.lr * args.lr_decay3}
    ], momentum=0.9, weight_decay=1e-3, nesterov=True)
    optimizer = op_copy(optimizer)

    dset_loaders = data_load(args)
    history = []
    soft_pseudo_label, coeff, roc_auc = evaluation(dset_loaders["test"], netF, netB, netC, args)
    history.append(roc_auc)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num, iter_test = 0, iter(dset_loaders["target"])

    while iter_num < max_iter:
        try:
            data_batch = next(iter_test)
        except StopIteration:
            iter_test = iter(dset_loaders["target"])
            data_batch = next(iter_test)

        if data_batch.num_graphs == 1: continue
        data_batch = data_batch.to(device)
        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)

        tar_idx = data_batch.local_sample_idx
        loss = feature_mixup(netF(data_batch), coeff[tar_idx].to(device), soft_pseudo_label[tar_idx].to(device), netB,
                             netC, args)
        if iter_num < args.warm * interval_iter + 1: loss *= 1e-6

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            soft_pseudo_label, coeff, roc_auc = evaluation(dset_loaders["test"], netF, netB, netC, args)
            history.append(roc_auc)

    print('\nROC-AUC history : {}\n'.format(history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CoWA for OGB Graphs')
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--dset', type=str, default='ogbg-molhiv')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--warm', type=float, default=0.0)
    parser.add_argument('--coeff', type=str, default='LPG')
    parser.add_argument('--lr_gamma', type=float, default=10.0)
    parser.add_argument('--lr_power', type=float, default=0.75)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--layer', type=str, default="wn")
    parser.add_argument('--classifier', type=str, default="bn")
    parser.add_argument('--distance', type=str, default='cosine')
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--output_src', type=str, default='pretrained_models')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset in ['ogbg-molhiv', 'ogbg-bbbp', 'ogbg-bace']:
        args.class_num = 2
    else:
        args.class_num = 2

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.output_dir = osp.join(args.output, args.dset)
    args.output_dir_src = osp.join(args.output_src, args.dset)
    os.makedirs(args.output_dir, exist_ok=True)
    args.prefix = '{}_alpha{}_lr{}_seed{}'.format(args.coeff, args.alpha, args.lr, args.seed)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.prefix + '.txt'), 'w')

    train_target(args)