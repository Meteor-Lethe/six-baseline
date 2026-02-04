# coding=utf-8
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from scipy.spatial.distance import cdist
import argparse
import time

# --- 1. 引入 GNN 模型 ---
from net.gnn_models import OGBGNN_Backbone, GNN_Classifier

# --- 2. 引入 OGB 和 PyG 依赖 ---
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.loader import DataLoader

# --- 3. 引入 CPGA 原有的 Loss 工具 ---
from utils import log, elr_loss, entropy_loss, ls_distance


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

class infoNCE_Fixed():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=300):
        self.features = features
        self.labels = labels
        self.class_num = class_num

    def get_posAndneg(self, features, labels, tgt_label=None, feature_q_idx=None, co_fea=None):
        self.features = features
        self.labels = labels
        device = features.device

        q_label = tgt_label[feature_q_idx]
        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = co_fea.unsqueeze(0)

        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.Tensor([]).to(device)

        if len(negative_sample_idx) > 0:
            for i in range(self.class_num - 1):
                negative_pairs = torch.cat(
                    (negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))
        else:
            negative_pairs = self.features[:self.class_num - 1]

        if negative_pairs.shape[0] < self.class_num - 1:
            diff = (self.class_num - 1) - negative_pairs.shape[0]
            if diff > 0 and len(negative_sample_idx) > 0:
                pads = torch.stack([self.features[random.choice(negative_sample_idx)] for _ in range(diff)])
                features_neg = torch.cat((negative_pairs, pads))
            else:
                features_neg = torch.cat([negative_pairs] + [feature_pos for _ in range(diff)])
        else:
            features_neg = negative_pairs

        return torch.cat((feature_pos, features_neg))


class infoNCE_g_Fixed():
    def __init__(self, features=None, labels=None, class_num=10, feature_dim=300):
        self.features = features
        self.labels = labels
        self.class_num = class_num

    def get_posAndneg(self, features, labels, feature_q_idx=None):
        self.features = features
        self.labels = labels
        device = features.device

        q_label = self.labels[feature_q_idx]

        positive_sample_idx = []
        for i, label in enumerate(self.labels):
            if label == q_label and i != feature_q_idx:
                positive_sample_idx.append(i)

        if len(positive_sample_idx) != 0:
            feature_pos = self.features[random.choice(positive_sample_idx)].unsqueeze(0)
        else:
            feature_pos = self.features[feature_q_idx].unsqueeze(0)

        negative_sample_idx = []
        for idx in range(features.shape[0]):
            if self.labels[idx] != q_label:
                negative_sample_idx.append(idx)

        negative_pairs = torch.tensor([]).to(device)

        if len(negative_sample_idx) > 0:
            for i in range(self.class_num - 1):
                negative_pairs = torch.cat(
                    (negative_pairs, self.features[random.choice(negative_sample_idx)].unsqueeze(0)))

        current_len = negative_pairs.shape[0]
        target_len = self.class_num - 1

        if current_len < target_len:
            if len(negative_sample_idx) > 0:
                makeup = [self.features[random.choice(negative_sample_idx)].unsqueeze(0) for _ in
                          range(target_len - current_len)]
                negative_pairs = torch.cat([negative_pairs] + makeup)
            else:
                makeup = [feature_pos for _ in range(target_len - current_len)]
                negative_pairs = torch.cat([negative_pairs] + makeup)

        return torch.cat((feature_pos, negative_pairs))


class OGBWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        label = data.y.squeeze().long()
        return data, label, index

    def __len__(self):
        return len(self.dataset)


class FeatureGenerator(nn.Module):
    def __init__(self, input_dim=100, feature_dim=300, num_classes=2):
        super(FeatureGenerator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, z, label):
        label_vec = self.label_emb(label)
        x = torch.cat((z, label_vec), dim=1)
        return self.mlp(x)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(SimpleMLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, p=None):
        return self.layer(x)


class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        super(LinearAverage, self).__init__()
        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / np.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = torch.mm(x, self.memory.t()) / self.params[0]
        return out

    def update_weight(self, features, index):
        # [关键修复] 1. 将 features 切断梯度，防止梯度反传到 Memory Update
        features = features.detach()

        if not torch.is_tensor(index):
            index = torch.tensor(index).long()
        index = index.to(features.device)

        # [关键修复] 2. 移除 .resize_as_
        weight_pos = self.memory.index_select(0, index.view(-1))

        # 进行动量更新
        weight_pos.mul_(self.params[1]).add_(features.mul(1 - self.params[1]))

        # 归一化
        norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(norm)

        # 写回内存
        self.memory.index_copy_(0, index, updated_weight)


# ==========================================
# 主 Trainer 类
# ==========================================

def arg_parser():
    parser = argparse.ArgumentParser()
    # 修改：默认使用 GPU 0
    parser.add_argument('--gpu', default='0', type=str, help='cpu or 0,1')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--generator_epoch', default=20, type=int)
    # 修改：source_model_path 可以为空，为空时根据 dataset_name 自动生成
    parser.add_argument('--source_model_path', default='', type=str)
    # 修改：支持选择数据集
    parser.add_argument('--dataset_name', default='ogbg-molhiv', type=str)
    # 对于 ogbg-molhiv/bace/bbbp，num_class 都是 2（虽然是 num_tasks=1 的二分类，但在 pytorch 中通常作为 2 类处理）
    parser.add_argument('--num_class', default=2, type=int)
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 修改：适配 GPU 设置
        if self.args.gpu != 'cpu' and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.gpu}')
            print(f"Running on GPU: cuda:{self.args.gpu}")
        else:
            self.device = torch.device('cpu')
            print("Running on CPU")

        self.loss_ce = nn.CrossEntropyLoss().to(self.device)
        self.loss_entropy = entropy_loss().to(self.device)

        self.infonce = infoNCE_Fixed(class_num=args.num_class, feature_dim=300)
        self.gen_c = infoNCE_g_Fixed(class_num=args.num_class, feature_dim=300)

        self.writer = SummaryWriter()
        self.alpha = 1
        self.logger = log()
        self.lr = args.lr
        self.same_ind = np.array([])
        self.confi_pre = np.array([])

    def cosine_similarity(self, feature, pairs):
        feature = F.normalize(feature)
        pairs = F.normalize(pairs)
        similarity = feature.mm(pairs.t())
        return similarity

    def exp_lr_scheduler(self, optimizer, step, lr_decay_step=2000, step_decay_weight=0.95):
        current_lr = self.lr * (step_decay_weight ** (step / lr_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        return optimizer

    # --- obtain_label ---
    def obtain_label(self, loader, my_net):
        my_net.eval()
        start_test = True
        with torch.no_grad():
            for data in loader:
                inputs, labels, t_indx = data
                # 修改：移动到 device
                inputs = inputs.to(self.device)
                outputs, feas = my_net(inputs)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_indx = t_indx.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    all_indx = torch.cat((all_indx, t_indx.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        max_prob, predict = torch.max(all_output, 1)

        model_ind = torch.squeeze((max_prob > self.alpha).nonzero())
        if model_ind.ndim == 0: model_ind = model_ind.unsqueeze(0)
        model_ind = model_ind.numpy()
        model_pre = predict.numpy().astype('int')

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        same_ind = np.where(pred_label.astype('int') == model_pre)[0]
        union = np.intersect1d(model_ind, same_ind)

        return union, pred_label.astype('int')[union]

    # --- obtain_residue_label ---
    def obtain_residue_label(self, loader, my_net, confi_pre, confi_ind):
        my_net.eval()
        start_test = True
        with torch.no_grad():
            for data in loader:
                inputs, labels, t_indx = data
                # 修改：移动到 device
                inputs = inputs.to(self.device)
                outputs, feas = my_net(inputs)

                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    all_indx = t_indx.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
                    all_indx = torch.cat((all_indx, t_indx.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc_output = aff.transpose().dot(all_fea)
        initc_output = initc_output / (1e-8 + aff.sum(axis=0)[:, None])

        class_tup = []
        for i in confi_pre:
            if i not in class_tup:
                class_tup.append(i)

        aff_confi = np.eye(K)[confi_pre]
        if len(confi_ind) > 0:
            initc = aff_confi.transpose().dot(all_fea[confi_ind])
        else:
            initc = np.zeros((K, all_fea.shape[1]))

        for i in range(self.args.num_class):
            if i not in class_tup:
                initc[i] = initc_output[i]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        max_prob, _ = torch.max(F.softmax(torch.from_numpy(1 - dd) / 0.07, dim=1), dim=1)
        pred_label = pred_label.astype('int')
        if len(confi_ind) > 0:
            pred_label[confi_ind] = confi_pre

        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        return pred_label, acc, all_indx.numpy(), max_prob.detach()

    # --- adaptation_step ---
    def adaptation_step(self, tgt_img, tgt_pre_label, sor_img, labels, t_indx, model, discriminator, fea_contrastor,
                        optimizer, epoch, sam_confidence):
        model.train()
        discriminator.train()
        fea_contrastor.train()
        optimizer.zero_grad()

        outputs, feas = model(tgt_img)

        reflect_fea = fea_contrastor(feas)
        all_fea = feas.float().cpu()
        all_ref_fea = reflect_fea.float().cpu()

        all_sam_indx, all_in, _ = np.intersect1d(t_indx, t_indx, return_indices=True)
        feat_t = F.normalize(all_fea.to(self.device))
        t_indx_gpu = torch.from_numpy(t_indx).long().to(self.device)

        feat_mat = self.lemniscate(feat_t, t_indx_gpu)
        feat_mat[:, t_indx_gpu] = -1 / 0.05
        feat_mat2 = torch.matmul(feat_t, feat_t.t()) / 0.05
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).type(torch.bool).to(self.device)
        feat_mat2.masked_fill_(mask, -1 / 0.05)
        loss_nc = 0.05 * self.loss_entropy(torch.cat([feat_mat, feat_mat2], 1))

        # 修改：确保初始化 Tensor 在正确的设备上
        adv_loss = torch.tensor(0.).to(self.device)
        source_dann = torch.tensor(0.).to(self.device)
        self.warm_epoch = 5
        if (epoch - self.generator_epoch) < self.warm_epoch:
            adv_loss = ls_distance(discriminator(all_fea.to(self.device)), 'target')
            source_dann = ls_distance(discriminator(sor_img), 'source')

        sor_img_con = fea_contrastor(sor_img)
        total_contrastive_loss = torch.tensor(0.).to(self.device)
        contrastive_label = torch.tensor([0]).to(self.device)
        gamma = 0.07
        nll = nn.NLLLoss()

        if len(all_in) > 0:
            for idx in range(len(all_in)):
                labels_tensor = labels if torch.is_tensor(labels) else torch.from_numpy(labels).long().to(self.device)

                pairs4q = self.infonce.get_posAndneg(features=sor_img_con, labels=labels_tensor,
                                                     tgt_label=tgt_pre_label,
                                                     feature_q_idx=t_indx[all_in[idx]],
                                                     co_fea=all_ref_fea[all_in[idx]].to(self.device))
                result = self.cosine_similarity(all_ref_fea[all_in[idx]].unsqueeze(0).to(self.device), pairs4q)
                numerator = torch.exp((result[0][0]) / gamma)
                denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                contrastive_loss = nll(result, contrastive_label) * sam_confidence[t_indx[all_in[idx]]]
                total_contrastive_loss = total_contrastive_loss + contrastive_loss
            total_contrastive_loss = total_contrastive_loss / len(all_in)

        all_class_prototypes = torch.Tensor([]).to(self.device)
        la_tup = []
        labels_cpu = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        for i, lab_id in enumerate(labels_cpu):
            if lab_id not in la_tup:
                la_tup.append(lab_id)
                all_class_prototypes = torch.cat((all_class_prototypes, sor_img_con[i].unsqueeze(0)))

        elr_loss_val = torch.tensor(0.).to(self.device)
        if len(all_class_prototypes) >= 2:
            if len(all_in) > 0:
                similarity_output = self.cosine_similarity(all_ref_fea[all_in].to(self.device),
                                                           all_class_prototypes) / gamma
                elr_loss_val = self.elr_loss(index=t_indx_gpu[all_in], output=similarity_output,
                                             label=torch.from_numpy(tgt_pre_label[t_indx[all_in]]).long().to(
                                                 self.device),
                                             contrastive_loss=total_contrastive_loss,
                                             confi_weight=sam_confidence[t_indx[all_in]])

        if (epoch - self.generator_epoch) < self.warm_epoch:
            loss = adv_loss + source_dann
        else:
            loss = elr_loss_val + loss_nc

        if loss != 0:
            loss.backward()
            optimizer.step()

        self.lemniscate.update_weight(feat_t, t_indx_gpu)
        return loss.item(), total_contrastive_loss.item()

    def obtain_pseudo_label_and_confidence_weight(self, test_loader, source_net):
        self.same_ind, self.confi_pre = self.obtain_label(test_loader, source_net)
        pseudo_label, pseudo_label_acc, all_indx, confidence_weight = self.obtain_residue_label(test_loader, source_net,
                                                                                                self.confi_pre,
                                                                                                self.same_ind)
        return pseudo_label, pseudo_label_acc, all_indx, confidence_weight

    def train_prototype_generator(self, epoch, batch_size_g, num_cls, optimizer_g, generator, source_classifier,
                                  loss_gen_ce):
        z = Variable(torch.rand(batch_size_g, 100)).to(self.device)
        labels = Variable(torch.randint(0, num_cls, (batch_size_g,))).to(self.device)

        features = generator(z, labels)
        output_teacher_batch = source_classifier(features)

        loss_one_hot = loss_gen_ce(output_teacher_batch, labels)

        total_contrastive_loss = torch.tensor(0.).to(self.device)
        if epoch >= 5:
            contrastive_label = torch.tensor([0]).to(self.device)
            margin = 0.5
            gamma = 1
            nll = nn.NLLLoss()
            for idx in range(features.size(0)):
                pairs4q = self.gen_c.get_posAndneg(features=features, labels=labels, feature_q_idx=idx)
                result = self.cosine_similarity(features[idx].unsqueeze(0), pairs4q)
                numerator = torch.exp((result[0][0] - margin) / gamma)
                denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                contrastive_loss = nll(result, contrastive_label)
                total_contrastive_loss = total_contrastive_loss + contrastive_loss
            total_contrastive_loss = total_contrastive_loss / features.size(0)

        optimizer_g.zero_grad()
        loss_G = loss_one_hot + total_contrastive_loss
        loss_G.backward()
        optimizer_g = self.exp_lr_scheduler(optimizer=optimizer_g, step=epoch)
        optimizer_g.step()

    def train(self):
        print(f"Adapting {self.args.dataset_name} using CPGA logic...")
        print(f"Device: {self.device}")

        raw_dataset = PygGraphPropPredDataset(name=self.args.dataset_name, root='dataset/')
        split_idx = raw_dataset.get_idx_split()

        dataset_target = OGBWithIndex(raw_dataset[split_idx["test"]])
        # Windows 下 num_workers 0
        dataloader_train = DataLoader(dataset_target, batch_size=self.args.batchsize, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset_target, batch_size=self.args.batchsize, shuffle=False, num_workers=0)

        print("Loading GNN model...")
        backbone = OGBGNN_Backbone(num_layers=5, emb_dim=300, dropout=0.5)
        source_net = GNN_Classifier(backbone, num_classes=self.args.num_class)

        # 修改：自动处理模型路径
        model_path = self.args.source_model_path
        if not model_path or not os.path.exists(model_path):
            auto_path = f'./model_source_gnn/best_gnn_model_{self.args.dataset_name}.pth'
            print(f"Specified model path not found or empty. Trying automatic path: {auto_path}")
            model_path = auto_path

        if os.path.exists(model_path):
            print(f"Loading checkpoint: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 适配 1-dim weights 到 2-class Softmax 格式 (兼容某些单任务模型输出)
            if 'fc.weight' in checkpoint and checkpoint['fc.weight'].shape[0] == 1 and self.args.num_class == 2:
                print("Adapting 1-dim weights to 2-class Softmax format...")
                old_weight = checkpoint['fc.weight']  # [1, 300]
                new_weight = torch.zeros((2, old_weight.shape[1]), device=self.device)
                new_weight[1] = old_weight[0]

                old_bias = checkpoint['fc.bias']  # [1]
                new_bias = torch.zeros((2,), device=self.device)
                new_bias[1] = old_bias[0]

                checkpoint['fc.weight'] = new_weight
                checkpoint['fc.bias'] = new_bias

            source_net.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run train_source_ogb.py first!")

        # 修改：将模型移动到 device
        source_net.to(self.device)

        source_classifier = source_net.fc

        feat_dim = 300
        generator = FeatureGenerator(feature_dim=feat_dim, num_classes=self.args.num_class).to(self.device)
        discriminator = SimpleMLP(input_dim=feat_dim, output_dim=1).to(self.device)
        fea_contrastor = SimpleMLP(input_dim=feat_dim, output_dim=feat_dim).to(self.device)

        self.lemniscate = LinearAverage(feat_dim, len(dataset_target), 0.05, 0.0).to(self.device)
        self.elr_loss = elr_loss(num_examp=len(dataset_target), num_classes=self.args.num_class).to(self.device)

        optimizer = optim.SGD(list(source_net.backbone.parameters()) +
                              list(discriminator.parameters()) +
                              list(fea_contrastor.parameters()),
                              lr=self.lr, momentum=0.9)
        optimizer_g = optim.SGD(generator.parameters(), lr=self.lr, momentum=0.9)
        loss_gen_ce = nn.CrossEntropyLoss().to(self.device)

        self.generator_epoch = self.args.generator_epoch
        n_epoch = self.args.max_epoch
        batch_size_g = self.args.batchsize * 2
        num_cls = self.args.num_class
        current_step = 0

        for epoch in range(n_epoch):
            if epoch < self.generator_epoch:
                generator.train()
                for _ in range(50):
                    self.train_prototype_generator(epoch, batch_size_g, num_cls, optimizer_g, generator,
                                                   source_classifier, loss_gen_ce)
                print(f"Epoch {epoch}: Generator training...")

            if epoch >= self.generator_epoch:
                generator.eval()

                z = Variable(torch.rand(num_cls * 50, 100)).to(self.device)
                labels_g = torch.cat([torch.full((50,), c) for c in range(num_cls)]).long().to(self.device)
                images = generator(z, labels_g).detach()

                self.alpha = 0.9 - (epoch - self.generator_epoch) / (n_epoch - self.generator_epoch) * 0.2

                print(f"Epoch {epoch}: Obtaining pseudo labels...")
                pseudo_label, pseudo_label_acc, all_indx, confidence_weight = self.obtain_pseudo_label_and_confidence_weight(
                    test_loader, source_net)

                print(f"Epoch {epoch}: Adaptation step...")
                for i, data in enumerate(dataloader_train):
                    s_img, s_label, s_indx = data
                    # 修改：数据移动到 device
                    s_img = s_img.to(self.device)

                    optimizer = self.exp_lr_scheduler(optimizer, current_step)

                    loss, contrastive_loss = self.adaptation_step(
                        tgt_img=s_img,
                        tgt_pre_label=pseudo_label,
                        sor_img=images,
                        labels=labels_g,
                        t_indx=s_indx.numpy(),
                        model=source_net,
                        discriminator=discriminator,
                        fea_contrastor=fea_contrastor,
                        optimizer=optimizer,
                        epoch=epoch,
                        sam_confidence=confidence_weight.float()
                    )
                    current_step += 1

                print(f"Epoch {epoch} Done. Loss: {loss:.4f}, Con_Loss: {contrastive_loss:.4f}")

                # 评估
                evaluator = Evaluator(self.args.dataset_name)
                source_net.eval()
                y_true, y_pred = [], []
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels, _ = data
                        logits, _ = source_net(inputs.to(self.device))
                        y_true.append(labels.view(-1, 1).numpy())

                        # 取出正类概率 (通常是 index 1)
                        probs = F.softmax(logits, dim=1)[:, 1].cpu().view(-1, 1).numpy()
                        y_pred.append(probs)

                auc = evaluator.eval({"y_true": np.vstack(y_true), "y_pred": np.vstack(y_pred)})['rocauc']
                print(f"Target AUC: {auc:.4f}")


if __name__ == '__main__':
    args = arg_parser()
    trainer = Trainer(args)
    trainer.train()