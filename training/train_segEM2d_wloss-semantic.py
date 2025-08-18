import gc
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.utils.dataloader_ninanjie import load_semantic_dataset
from utils.dataloader_ninanjie import collate_fn_2D_fib25_Train


## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &


def set_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


####ACRLSD模型
class ACRLSD(torch.nn.Module):
    def __init__(
            self,
    ):
        super(ACRLSD, self).__init__()

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet2d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.lsd_predict = torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') # 去掉 'module.' 前缀
        new_state_dict[name] = v
    return new_state_dict


class segEM2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/'
                      'training/output/checkpoints/ACRLSD_2D(ninanjie)_semantic_Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=4,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.class_predict = torch.nn.Conv2d(in_channels=12, out_channels=4, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_raw, y_affinity.detach(), x_prompt.unsqueeze(1)], dim=1)

        y_logits = self.class_predict(self.model_mask(y_concat))

        return y_logits, y_lsds, y_affinity


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_probs, target_mask):
        """
            多类别 Dice损失：计算每个类别的 Dice 系数，取平均
            pred_probs: [B, C, H, W] 经过 softmax 的概率图
            target_mask: [B, H, W] 类别索引（0-C-1）
        """
        total_dice = 0.0
        num_classes = pred_probs.shape[1]
        for c in range(num_classes):
            # 提取当前类别的预测概率和目标掩码
            pred = pred_probs[:, c, ...]  # [B, H, W]
            target = (target_mask == c).float()  # [B, H, W] 二值化当前类

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_dice += dice
        return 1 - (total_dice / num_classes)
class WeightedDiceLoss(torch.nn.Module):
    def __init__(self, weights, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.weights = weights  # 类别权重，形状为 [C]，如 [0.1, 0.2, 0.3, 0.4]

    def forward(self, pred_probs, target_mask):
        """
            加权多类别 Dice损失：按类别权重计算Dice系数的加权和
            pred_probs: [B, C, H, W] 经过 softmax 的概率图
            target_mask: [B, C, H, W] 类别索引
        """
        total_dice = 0.0
        num_classes = pred_probs.shape[1]

        for c in range(num_classes):
            pred = pred_probs[:, c, ...]
            target = target_mask[:, c, ...]

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_dice += dice * self.weights[c]  # 乘以类别权重

        # 加权平均（除以权重和，避免权重总和影响损失量级）
        return 1 - (total_dice / sum(self.weights))

class WeightedAffinityLoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred_probs, gt_affinity):
        num_classes = pred_probs.shape[1]
        base_mat = torch.zeros_like(gt_affinity)
        for i in range(num_classes):
            base_mat += pred_probs[:, i, ...].unsqueeze(1) * gt_affinity
        edge_probs = torch.sum(base_mat)
        return edge_probs / torch.sum(gt_affinity)


def model_step(model, optimizer, input_image, input_prompt, gt_semantic_mask, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_probs, y_lsds, y_affinity = model(input_image, input_prompt)


    loss1 = F.cross_entropy(y_probs, gt_semantic_mask.long())
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(y_probs, gt_semantic_mask)

    # loss = loss1 + loss2
    weight = torch.tensor([0.1, 0.3, 0.3, 0.3], device=y_probs.device)
    weight = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    y_probs_avg = (y_probs * weight).sum(dim=1, keepdim=True)
    loss3 = torch.sum(y_probs_avg * gt_affinity) / torch.sum(gt_affinity)
    loss = loss1 * WEIGHT_LOSSES[0] + loss2 * WEIGHT_LOSSES[1] + loss3 * WEIGHT_LOSSES[2]

    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    else:
        y_probs = torch.argmax(y_probs, dim=1)

    return loss, y_probs


def weighted_model_step(model, optimizer, input_image, input_prompt, gt_semantic_mask, gt_affinity,
                        activation, pos_weight, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_logits, y_lsds, y_affinity = model(input_image, input_prompt)
    y_probs = activation(y_logits)

    BCE_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.view(1, 4, 1, 1)).to(device)
    Diceloss_fn = WeightedDiceLoss(pos_weight).to(device)
    Affinityloss_fn = WeightedAffinityLoss(pos_weight).to(device)


    losses = [
        BCE_fn(y_logits, gt_semantic_mask),
        Diceloss_fn(y_probs, gt_semantic_mask),
        Affinityloss_fn(y_probs, gt_affinity)
    ]

    loss = losses[0] * WEIGHT_LOSSES[0] + losses[1] * WEIGHT_LOSSES[1] + losses[2] * WEIGHT_LOSSES[2]

    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    else:
        y_probs = torch.argmax(y_probs, dim=1)

    return loss, y_probs, losses


def calculate_multi_class_voi(pred, gt):
    """
    多类别VOI计算（GPU加速版）
    参数:
        pred: 预测标签张量，形状[H, W]，值为0-3（可在CPU/GPU，函数内部自动转移到GPU）
        gt: 真实标签张量，形状[H, W]，值为0-3（同上）
    返回:
        voi: 计算得到的VOI值（标量）
    """
    # 转换为GPU张量并展平（若已在GPU则不重复转移）
    pred = torch.as_tensor(pred, dtype=torch.long).flatten()
    gt = torch.as_tensor(gt, dtype=torch.long).flatten()
    total = pred.numel()  # 总像素数
    if total == 0:
        return 0.0

    # ----------------------
    # 1. 计算边缘概率分布 P(pred) 和 P(gt)
    # ----------------------
    def get_prob(tensor):
        # 计算唯一值及对应计数（GPU上执行）
        unique, counts = torch.unique(tensor, return_counts=True)
        prob = counts.float() / total
        return unique, prob

    # 预测标签的概率分布
    pred_unique, prob_pred = get_prob(pred)
    # 真实标签的概率分布
    gt_unique, prob_gt = get_prob(gt)

    # ----------------------
    # 2. 计算联合概率分布 P(pred, gt)
    # ----------------------
    # 拼接预测和真实标签为二维张量（形状[N, 2]），用于计算联合唯一值
    joint = torch.stack([pred, gt], dim=1)  # [N, 2]
    # 计算联合唯一值及计数（关键优化：替代循环构建字典）
    joint_unique, joint_counts = torch.unique(joint, dim=0, return_counts=True)
    joint_prob = joint_counts.float() / total  # 联合概率

    # ----------------------
    # 3. 计算熵 H(pred) 和 H(gt)
    # ----------------------
    eps = 1e-10  # 避免log2(0)
    h_pred = -torch.sum(prob_pred * torch.log2(prob_pred + eps))  # 向量化计算
    h_gt = -torch.sum(prob_gt * torch.log2(prob_gt + eps))

    # ----------------------
    # 4. 计算互信息 I(pred, gt)
    # ----------------------
    mi = 0.0
    # 遍历所有联合唯一值（数量远少于总像素，高效）
    for idx in range(joint_unique.shape[0]):
        p_val = joint_unique[idx, 0]  # 预测值
        g_val = joint_unique[idx, 1]  # 真实值
        p_joint = joint_prob[idx]      # 联合概率

        # 找到对应边缘概率（利用张量索引加速）
        p_p = prob_pred[pred_unique == p_val].item()
        p_g = prob_gt[gt_unique == g_val].item()

        if p_p > eps and p_g > eps and p_joint > eps:
            mi += p_joint * torch.log2(p_joint / (p_p * p_g) + eps).item()

    # 计算VOI
    voi = (h_pred + h_gt - 2 * mi).item()
    return voi

def trainer():
    model = segEM2d().to(device)
    # set device
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)  # 并行使用两块
    # model = torch.nn.DataParallel(model)  # 默认使用所有的 device_ids
    # model = model.to(device)

    save_dir = os.path.join('./output/segEM2d(ninanjie)_semantic',
                            f'{"-".join([str(_) for _ in WEIGHT_LOSSES])}',
                            f'{"-".join([str(_) for _ in label_weights])}')
    os.makedirs(save_dir, exist_ok=True)
    Save_Name = f'segEM2d(ninanjie)_semantic-{crop_size}'

    ## 创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logfile = f'{save_dir}/log_{Save_Name}.txt'
    csvfile = f'{save_dir}/log_{Save_Name}.csv'
    fh = logging.FileHandler(logfile, mode='a', delay=False)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f'''Starting training:
        training_epochs:  {training_epochs}
        Num_slices_train: {len(train_dataset)}
        Num_slices_val:   {len(val_dataset)}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Device:           {device.type}
        ''')

    ##开始训练
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # set activation
    activation = torch.nn.Sigmoid().to(device)
    pos_weight = torch.tensor(label_weights, device=device, dtype=torch.float32)

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 10000
    Best_voi = 10000
    Best_epoch = 0
    early_stop_count = 64
    no_improve_count = 0
    with (tqdm(total=training_epochs) as pbar):
        analysis = pd.DataFrame(columns=['loss', 'voi', 'bce', 'dice', 'affinity'])
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity in train_loader:
                weighted_model_step(model, optimizer, raw, point_map, labels, gt_affinity, activation,
                                    pos_weight=pos_weight, train_step=True)
                count += 1
                progress_desc = f'Train: {count}/{total}, '
                train_desc = f"Best: {Best_epoch}, loss: {Best_val_loss:.2f}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            acc_loss = []
            count, total = 0, len(val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            judgment_rates = 0.0
            # metrics = np.asarray((0, 0, 0, 0), dtype=np.float64)
            for raw, labels, point_map, mask, gt_affinity in val_loader:
                with torch.no_grad():
                    loss_value, y_pred, losses = weighted_model_step(model, optimizer, raw, point_map, labels,
                                                                     gt_affinity, activation,
                                                                     pos_weight=pos_weight, train_step=False)
                    labels = torch.argmax(labels, dim=1)
                    judgment_rates += calculate_multi_class_voi(y_pred, labels)
                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)
                acc_loss.append(loss_value)
            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            judgment_rates /= (total + 0.)
            if epoch > 32 and judgment_rates < Best_voi:
                Best_voi = judgment_rates
                torch.save(model.module.state_dict(), f'{save_dir}/{Save_Name}_voi.model')

            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()
            analysis.loc[len(analysis)] = [val_loss, judgment_rates].extend(losses)
            val_desc = f'Best VOI: {Best_voi:.4f}, VOI: {judgment_rates:.4f}' if epoch > 32 else 'Updating VOI'

            epoch += 1
            pbar.update(1)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(), f'{save_dir}/{Save_Name}.model')
                no_improve_count = 0
            else:
                no_improve_count += 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f}, voi = {:.6f} with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, judgment_rates, Best_val_loss, Best_epoch))
            fh.flush()
            # writer.add_scalar('val_loss', val_loss, epoch)
            analysis.to_csv(csvfile)

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break

    torch.save(model.module.state_dict(), f'{save_dir}/{Save_Name}_final.model')
    del model, optimizer, activation
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 2e-4
    batch_size = 48

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,4,5'  # 设置所有可以使用的显卡，共计四块,
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]   # 选中显卡
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ##装载数据
    dataset_names = ['second_6', 'fourth_1']

    ##装载数据
    # WEIGHT_LOSSES = [1., 2., 1.]
    # label_weights = [1, 16, 10, 50]
    crop_size = 576
    crop_xyz = [4, 4, 1]
    train_dataset, val_dataset = [], []
    for dataset_name in dataset_names:
        for i in range(crop_xyz[0] - 1):
            for j in range(crop_xyz[1]):
                for k in range(crop_xyz[2]):
                    train_tmp, val_tmp = load_semantic_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                                               require_xz_yz=False, from_temp=True,
                                                               crop_xyz=crop_xyz, chunk_position=[i, j, k],
                                                               crop_size=crop_size)
                    train_dataset.append(train_tmp)
                    val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=48, pin_memory=True,
                            drop_last=False, collate_fn=collate_fn_2D_fib25_Train)


    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
        for raw, labels, point_map, mask, gt_affinity, _ in tqdm(tmp_loader, leave=False, desc='load to cuda'):
            ##Get Tensor
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            labels = torch.as_tensor(labels, dtype=torch.float, device=device)  # (batch, 1, height, width)
            point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, height, width)
            mask = torch.as_tensor(mask, dtype=torch.float, device=device)  # (batch, 1, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append([raw, labels, point_map, mask, gt_affinity])
        return res


    train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)

    for WEIGHT_LOSSES in [[1., 2., 1.], [1., 1., 2.], [2., 1., 1.], [1., 1., 1.]]:
        for label_weights in [[1, 16, 10, 50], [1, 8, 5, 25], [1, 4, 3, 7]]:
            save_dir = os.path.join('./output/segEM2d(ninanjie)_semantic',
                                    f'{"-".join([str(_) for _ in WEIGHT_LOSSES])}',
                                    f'{"-".join([str(_) for _ in label_weights])}')
            if os.path.exists(save_dir):
                continue
            os.makedirs(save_dir)
            trainer()

