# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
from tqdm import tqdm  # 引入进度条库

# 引入我们定义的 GNN 模型
from net.gnn_models import OGBGNN_Backbone, GNN_Classifier

# 引入 OGB 和 PyG 依赖
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

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

def arg_parser():
    parser = argparse.ArgumentParser()
    # 修改：默认使用 GPU 0，如果想用 CPU 可以传 'cpu'
    parser.add_argument('--gpu', default='0', type=str, help='GPU index (e.g. 0) or cpu')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    # 默认轮次，你可以通过命令行参数修改它
    parser.add_argument('--max_epoch', default=50, type=int)
    # 你可以通过命令行参数修改为 ogbg-molbbbp 或 ogbg-molbace
    parser.add_argument('--dataset_name', default='ogbg-molhiv', type=str)

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # OGBG-MolHiv 是二分类任务，使用 BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

        # 修改：适配 GPU 设备设置
        if self.args.gpu != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.gpu}')
            print(f"Running on GPU: cuda:{self.args.gpu}")
        else:
            self.device = torch.device('cpu')
            print("Running on CPU")

    def train_process(self, model, optimizer, batch_data):
        model.train()
        optimizer.zero_grad()

        # 1. 数据搬运到 GPU/CPU
        batch_data = batch_data.to(self.device)

        # 2. 前向传播
        logits, _ = model(batch_data)

        # 3. 处理标签
        y_true = batch_data.y.to(torch.float32)
        is_labeled = batch_data.y == batch_data.y

        # 计算 Loss
        loss = self.criterion(logits[is_labeled], y_true[is_labeled])

        # 4. 反向传播
        loss.backward()
        optimizer.step()

        return loss.item()

    def validate(self, model, loader, evaluator):
        model.eval()
        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for batch_data in loader:
                # 修改：数据搬运到 GPU/CPU
                batch_data = batch_data.to(self.device)
                logits, _ = model(batch_data)

                y_true_list.append(batch_data.y.view(logits.shape).detach().cpu())
                y_pred_list.append(logits.detach().cpu())

        y_true = torch.cat(y_true_list, dim=0).numpy()
        y_pred = torch.cat(y_pred_list, dim=0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        # 这里 Evaluator 会自动根据数据集名称调用对应的 Metric（对于 molhiv/bace/bbbp 都是 ROC-AUC）
        result_dict = evaluator.eval(input_dict)
        return result_dict['rocauc']

    def train(self):
        # 1. 准备数据集
        print(f"Loading dataset {self.args.dataset_name}...")
        dataset = PygGraphPropPredDataset(name=self.args.dataset_name, root='dataset/')

        # 2. 划分数据集
        split_idx = dataset.get_idx_split()

        # Windows 下 num_workers 必须设为 0，否则可能会报错或死锁
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=self.args.batchsize, shuffle=True,
                                  num_workers=0)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=self.args.batchsize, shuffle=False,
                                  num_workers=0)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=self.args.batchsize, shuffle=False,
                                 num_workers=0)

        # 3. 初始化模型
        num_tasks = dataset.num_tasks
        print(f"Initializing GNN model for {num_tasks} tasks...")

        backbone = OGBGNN_Backbone(num_layers=5, emb_dim=300, dropout=0.5)
        model = GNN_Classifier(backbone, num_classes=num_tasks)
        # 修改：模型搬运到 GPU/CPU
        model = model.to(self.device)

        # 4. 优化器
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        evaluator = Evaluator(self.args.dataset_name)

        # 5. 训练循环
        best_val_auc = 0
        model_save_dir = './model_source_gnn'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        print(f"Start Training for {self.args.max_epoch} epochs...")

        for epoch in range(self.args.max_epoch):
            start_time = time.time()
            loss_list = []

            # --- Training with tqdm ---
            # 使用 tqdm 包装数据加载器来显示进度条
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:03d}/{self.args.max_epoch}", unit="batch")

            for batch_data in pbar:
                loss = self.train_process(model, optimizer, batch_data)
                loss_list.append(loss)
                # 实时更新进度条右侧的 Loss 显示
                pbar.set_postfix({'loss': f"{loss:.4f}"})

            train_loss = np.mean(loss_list)

            # --- Validation ---
            # 为了保持控制台整洁，验证过程不加进度条，或者可以加简单的 print
            val_auc = self.validate(model, valid_loader, evaluator)
            test_auc = self.validate(model, test_loader, evaluator)

            epoch_time = time.time() - start_time

            # 打印本轮总结 (tqdm 会自动换行，这里打印在下一行)
            print(f"Epoch {epoch + 1:03d} Summary | Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f} | "
                  f"Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                save_path = os.path.join(model_save_dir, f'best_gnn_model_{self.args.dataset_name}.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved to {save_path}")

            print("-" * 60)

        print("Finished Training.")


if __name__ == '__main__':
    args = arg_parser()
    trainer = Trainer(args)
    trainer.train()