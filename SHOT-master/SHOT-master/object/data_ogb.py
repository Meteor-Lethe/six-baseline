# 文件名: SHOT-master/object/data_ogb.py
import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset

torch.load = lambda *args, **kwargs: torch.serialization.load(*args, **{**kwargs, "weights_only": False})


class IndexedDataset:
    """
    [修改原因] SHOT 的无监督算法（image_target.py）需要根据样本的 index 来更新伪标签。
    PyG 默认只返回 data，所以我们需要包装一下，让它返回 (data, label, index)。
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        # [实现细节] OGB 的 label 通常在 data.y，且形状为 (1, 1)，
        # 我们将其 squeeze 并转为 long 类型以适配 CrossEntropyLoss
        label = data.y.squeeze().long()
        return data, label, idx


def get_ogb_loaders(args):
    """
    [修改原因] 替代原有的 image_load 函数，专门处理图数据。
    [实现细节] 定义 Train set 为 Source, Test set 为 Target。
    """
    # 【新增】使用参数指定的数据集名称，支持 ogbg-molhiv, ogbg-bbbp, ogbg-bace 等
    dataset_name = args.dset
    print(f"Loading OGB Dataset: {dataset_name}")

    # 加载数据集
    dataset = PygGraphPropPredDataset(name=dataset_name, root='./data')
    split_idx = dataset.get_idx_split()

    # 划分源域和目标域
    # Source = Train Split
    source_dataset_raw = dataset[split_idx["train"]]
    # Target = Test Split
    target_dataset_raw = dataset[split_idx["test"]]

    # 包装索引
    source_dataset = IndexedDataset(source_dataset_raw)
    target_dataset = IndexedDataset(target_dataset_raw)

    # 构建 DataLoader
    # [实现细节] PyG 的 DataLoader 会自动将多个图 collate 成一个 Batch 对象
    train_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker
    )

    # Target 需要两个 loader：一个用于训练(shuffle)，一个用于生成标签(no shuffle)
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker
    )

    target_loader_test = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker
    )

    return train_loader, target_loader, target_loader_test