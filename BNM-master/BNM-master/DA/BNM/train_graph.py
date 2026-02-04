import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator  # 引入 Evaluator

# 引入修改后的图网络和辅助模块
import network_graph as network  #
import lr_schedule  #

# ==============================================================================
# 【修复 PyTorch 2.6+ 加载 OGB 数据集报错的问题】
# 保持这个补丁，防止加载 OGB 数据集时报错
# ==============================================================================
_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = patched_torch_load  #


# ==============================================================================

def graph_classification_test(loader, model, device, evaluator):
    """
    测试函数：计算 ROC-AUC (符合 OGB 官方标准)
    """
    model.eval()  #
    y_true = []
    y_pred = []

    with torch.no_grad():  #
        for batch in loader:
            batch = batch.to(device)  #
            # 模型输出 (features, logits)
            _, outputs = model(batch)  #

            # 获取正类的概率 (softmax 后的第1列)
            # molhiv/bbbp/bace 均为二分类任务
            pred = outputs.softmax(dim=1)[:, 1].view(-1, 1)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    # 修复报错点：将 y_score 改为 y_pred
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)

    return result_dict['rocauc']


def train(config):
    # 自动适配 GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")  #

    ## 1. 准备数据 (根据参数选择数据集)
    print(f"Loading {config['dataset_name']} Dataset...")
    dataset = PygGraphPropPredDataset(name=config['dataset_name'], root='dataset/')  #
    split_idx = dataset.get_idx_split()  #

    # 初始化官方评估器
    evaluator = Evaluator(name=config['dataset_name'])

    # 方案 A: Train -> Source, Valid -> Target
    source_dataset = dataset[split_idx["train"]]  #
    target_dataset = dataset[split_idx["valid"]]  #
    test_dataset = dataset[split_idx["test"]]  #

    train_bs = config["batch_size"]  #

    # 构建加载器
    source_loader = DataLoader(source_dataset, batch_size=train_bs, shuffle=True, drop_last=True)  #
    target_loader = DataLoader(target_dataset, batch_size=train_bs, shuffle=True, drop_last=True)  #
    test_loader = DataLoader(test_dataset, batch_size=train_bs, shuffle=False)  #

    dset_loaders = {
        "source": source_loader,
        "target": target_loader,
        "test": test_loader
    }  #

    ## 2. 设置网络
    # 初始化 GIN_BNM
    base_network = network.GIN_BNM(num_tasks=1, num_layers=5, emb_dim=300, dropout=0.5)  #
    base_network = base_network.to(device)  #

    ## 3. 设置优化器
    parameter_list = base_network.get_parameters()  #
    optimizer_config = config["optimizer"]  #
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))  #

    # 学习率调度器
    schedule_param = optimizer_config["lr_param"]  #
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]  #

    ## 4. 训练循环
    best_rocauc = 0.0

    # 创建迭代器
    iter_source = iter(dset_loaders["source"])  #
    iter_target = iter(dset_loaders["target"])  #

    for i in range(config["num_iterations"]):  #
        # Test Phase
        if i % config["test_interval"] == config["test_interval"] - 1:  #
            # 使用官方 ROC-AUC 评估
            temp_rocauc = graph_classification_test(dset_loaders["test"], base_network, device, evaluator)

            if temp_rocauc > best_rocauc:  #
                best_rocauc = temp_rocauc
                # 保存最佳模型
                torch.save(base_network.state_dict(), osp.join(config["output_path"], "best_model.pth"))  #

            log_str = "iter: {:05d}, ROC-AUC: {:.5f}, best: {:.5f}".format(i, temp_rocauc, best_rocauc)
            config["out_file"].write(log_str + "\n")  #
            config["out_file"].flush()  #
            print(log_str)  #

        # Train Phase
        base_network.train(True)  #
        optimizer = lr_scheduler(optimizer, i, **schedule_param)  #
        optimizer.zero_grad()  #

        # 数据加载处理 (Infinite Loop Logic)
        try:
            batch_source = next(iter_source)  #
        except StopIteration:
            iter_source = iter(dset_loaders["source"])  #
            batch_source = next(iter_source)  #

        try:
            batch_target = next(iter_target)  #
        except StopIteration:
            iter_target = iter(dset_loaders["target"])  #
            batch_target = next(iter_target)  #

        batch_source = batch_source.to(device)  #
        batch_target = batch_target.to(device)  #

        # Forward Pass
        _, outputs_source = base_network(batch_source)  #
        labels_source = batch_source.y.view(-1).long()  #
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)  #

        # Target flow: 计算 BNM Loss
        _, outputs_target = base_network(batch_target)  #
        softmax_tgt = nn.Softmax(dim=1)(outputs_target)  #

        # BNM Methods Implementation
        if config["method"] == "BNM":  #
            _, s_tgt, _ = torch.svd(softmax_tgt)  #
            transfer_loss = -torch.mean(s_tgt)  #
        elif config["method"] == "ENT":  #
            transfer_loss = -torch.mean(torch.sum(softmax_tgt * torch.log(softmax_tgt + 1e-8), dim=1))  #
        else:
            transfer_loss = torch.tensor(0.0).to(device)  #

        total_loss = config["trade_off"] * transfer_loss + classifier_loss  #

        # Backward
        total_loss.backward()  #
        optimizer.step()  #

        if i % config["print_num"] == 0:  #
            log_str = "iter: {:05d}, transfer_loss: {:.5f}, clf_loss: {:.5f}".format(
                i, transfer_loss.item(), classifier_loss.item())  #
            config["out_file"].write(log_str + "\n")  #
            config["out_file"].flush()  #
            print(log_str)  #

    return best_rocauc  #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Transfer Learning BNM')
    parser.add_argument('--dataset', type=str, default='ogbg-molhiv', help="ogbg-molhiv, ogbg-bbbp, ogbg-bace")
    parser.add_argument('--output_dir', type=str, default='san_graph', help="output directory")
    parser.add_argument('--method', type=str, default='BNM', help="Options: BNM, ENT")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="parameter for transfer loss")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--num_iterations', type=int, default=1000, help="number of iterations")
    parser.add_argument('--test_interval', type=int, default=100, help="test interval")
    parser.add_argument('--print_num', type=int, default=20, help="print interval")

    args = parser.parse_args()

    config = {}
    config["dataset_name"] = args.dataset
    config["method"] = args.method
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["test_interval"] = args.test_interval
    config["output_path"] = args.output_dir
    config["batch_size"] = args.batch_size
    config["trade_off"] = args.trade_off

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"], exist_ok=True)  #
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")  #

    config["optimizer"] = {
        "type": optim.Adam,  #
        "optim_params": {'lr': args.lr, "weight_decay": 0.0},  #
        "lr_type": "inv",  #
        "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}  #
    }

    print(f'Starting Graph BNM Training on {args.dataset}...')
    train(config)