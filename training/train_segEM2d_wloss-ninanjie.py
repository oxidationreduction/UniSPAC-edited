import logging
import os
import random
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.utils.dataloader_ninanjie import get_acc_prec_recall_f1
from skimage.metrics import variation_of_information
from utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train, load_train_dataset, collate_fn_2D_fib25_Train
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

WEIGHT_LOSS2 = 15
WEIGHT_LOSS3 = 1


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
                      'training/output/checkpoints/ACRLSD_2D(ninanjie)_subcell_1_Best_in_val.model')
        weights = torch.load(model_path, map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False

        # create our network, 2 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet2d(
            in_channels=3,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.mask_predict = torch.nn.Conv2d(in_channels=12, out_channels=1, kernel_size=1)  # 最终输出层的卷积操作

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_raw, x_prompt):
        y_lsds, y_affinity = self.model_affinity(x_raw)

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity.detach()], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, foreground_weight):
        # 仅关注前景的Dice损失（target=1的区域）
        pred_fg = pred * target  # 前景预测
        target_fg = target  # 前景标签

        # 加权计算交集和并集（前景权重更高）
        intersection = (pred_fg * target_fg * foreground_weight).sum()
        union = (pred_fg * foreground_weight).sum() + target_fg.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice  # 损失值越小越好


def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask, y_lsds, y_affinity = model(input_image, input_prompt)
    y_mask_ = y_mask.squeeze()
    gt_binary_mask_ = gt_binary_mask.squeeze()

    loss1 = F.binary_cross_entropy(y_mask_, gt_binary_mask_)
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(1 - y_mask_, 1 - gt_binary_mask_)

    # loss = loss1 + loss2

    loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)
    loss = loss1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    else:
        y_mask, gt_binary_mask = y_mask_, gt_binary_mask_

    return loss, y_mask


def weighted_model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    # lsd_logits,affinity_logits = model(raw)
    y_mask, y_lsds, y_affinity = model(input_image, input_prompt)
    y_mask_ = y_mask.squeeze()
    gt_binary_mask_ = gt_binary_mask.squeeze()

    foreground = gt_binary_mask_.sum()
    total = gt_binary_mask_.numel()
    background = total - foreground
    weight = background / foreground

    weights = torch.where(gt_binary_mask_ == 1, weight, torch.tensor(1.0, device=device))
    loss1 = F.binary_cross_entropy(y_mask_, gt_binary_mask_, weight=weights)
    Diceloss_fn = WeightedDiceLoss().to(device)
    loss2 = Diceloss_fn(1. - y_mask_, 1. - gt_binary_mask_, weights)

    # loss = loss1 + loss2

    loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)
    loss = loss1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    else:
        y_mask, gt_binary_mask = y_mask_, gt_binary_mask_

    return loss, y_mask


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 64

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5'  # 设置所有可以使用的显卡，共计四块,
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]   # 选中显卡
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = segEM2d().to(device)
    # set device
    model = nn.DataParallel(model, device_ids=device_ids, output_device=device)#并行使用两块
    # model = torch.nn.DataParallel(model)  # 默认使用所有的 device_ids
    # model = model.to(device)

    ##装载数据
    class_num = 1
    dataset_names = ['second_6']
    Save_Name = 'segEM2d(ninanjie)_subcell_{}-w2-{}-w3-{}_512'.format(class_num, WEIGHT_LOSS2, WEIGHT_LOSS3)

    ##装载数据
    crop_xyz = [4, 4, 1]
    train_dataset, val_dataset = [], []
    for dataset_name in dataset_names:
        for i in range(crop_xyz[0] - 1):
            for j in range(crop_xyz[1]):
                for k in range(crop_xyz[2]):
                    train_tmp, val_tmp = load_train_dataset(dataset_name, raw_dir='raw_2', label_dir='label_2',
                                                            from_temp=True, require_xz_yz=False, crop_size=512,
                                                            crop_xyz=crop_xyz, chunk_position=[i, j, k])
                    train_dataset.append(train_tmp)
                    val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=48, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=(batch_size // 2) + 1, shuffle=True, num_workers=48, pin_memory=True,
                            collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
        for raw, labels, point_map, mask, gt_affinity, _ in tqdm(tmp_loader, leave=True, desc='load to cuda'):
            ##Get Tensor
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)  # (batch, height, width)
            mask = torch.as_tensor(mask, dtype=torch.float, device=device)  # (batch, 1, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append([raw, labels, point_map, mask, gt_affinity])
        return res

    train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)

    ## 创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    csvfile = './output/log/log_{}.csv'.format(Save_Name)
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
    activation = torch.nn.Sigmoid()

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 10000
    Best_epoch = 0
    early_stop_count = 100
    no_improve_count = 0
    with (tqdm(total=training_epochs) as pbar):
        analysis = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1', 'voi'])
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
                loss_value, pred = weighted_model_step(model, optimizer, raw, point_map, mask, gt_affinity, activation,
                                              train_step=True)
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
            metrics = np.asarray((0, 0, 0, 0), dtype=np.float64)
            voi_val = 0
            for raw, labels, point_map, mask, gt_affinity in val_loader:
                with torch.no_grad():
                    loss_value, y_mask = weighted_model_step(model, optimizer, raw, point_map, mask, gt_affinity, activation,
                                               train_step=False)
                    binary_y_mask = ((np.asarray(y_mask.detach().cpu()) > 0.5) + 0).flatten()
                    binary_gt_seg = ((np.asarray(labels) > 0.0) + 0).flatten()

                    metrics += np.asarray(get_acc_prec_recall_f1(binary_y_mask, binary_gt_seg))
                    voi_val += np.sum(variation_of_information(binary_y_mask, binary_gt_seg))

                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            metrics /= (total + 0.0)
            voi_val /= (total + 0.0)
            acc, prec, recall, f1 = metrics[:]
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'
            analysis.loc[len(analysis)] = [acc, prec, recall, f1, voi_val]
            acc_loss.append(loss_value)
            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()

            epoch += 1
            pbar.update(1)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(), './output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count += 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            fh.flush()
            # writer.add_scalar('val_loss', val_loss, epoch)
            analysis.to_csv(csvfile)

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break