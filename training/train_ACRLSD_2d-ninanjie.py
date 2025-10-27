import gc
import itertools
import logging
import os
import random
import sys
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import variation_of_information
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# from torchmetrics import functional
from models.unet2d import UNet2d
from training.models.ACRLSD import ACRLSD_2D
from training.utils.aftercare import visualize_and_save_affinity, normalize_affinity
from training.utils.dataloader_ninanjie import load_train_dataset, collate_fn_2D_fib25_Train


# from utils.dataloader_cremi import Dataset_2D_cremi_Train,collate_fn_2D_cremi_Train

## CUDA_VISIBLE_DEVICES=1 python train_ACRLSD_2d.py &

def set_seed(seed=19260817):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


def model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
               scaler=None, train_step=True, auto_loss=False):
    if train_step:
        optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=scaler is not None):
        lsd_logits, affinity_logits = model(raw)
        loss1 = loss_fn(lsd_logits, gt_lsds)
        loss2 = loss_fn(affinity_logits, gt_affinity)
        if auto_loss:
            loss_value = loss1 / (loss1.detach().item() + 1e-5) + loss2 / (loss2.detach().item() + 1e-5)
        else:
            loss_value = loss1 + loss2

    if train_step:
        if scaler is not None:
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_value.backward()
            optimizer.step()

    lsd_output = activation(lsd_logits)
    affinity_output = activation(affinity_logits)

    outputs = {
        'pred_lsds': lsd_output,
        'lsds_logits': lsd_logits,
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }
    return loss_value, outputs


def weighted_model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
                        scaler=None, train_step=True, auto_loss=False):
    if train_step:
        optimizer.zero_grad()

    # 前向传播
    lsd_logits, affinity_logits = model(raw)

    # --------------------------
    # 1. 提取前景区域并计算权重
    # --------------------------
    # 假设亲和度图中前景相关区域为非零值（可根据实际标签定义调整）
    # 生成前景掩码（1表示前景相关区域，0表示背景）
    fg_mask_affinity = (gt_affinity > 0).float()  # 亲和度前景掩码
    fg_mask_lsds = (gt_lsds > 0).float()  # LSDS前景掩码（根据实际标签调整）

    # 计算前景/背景像素数量（亲和度）
    fg_pixels_affinity = fg_mask_affinity.sum()
    bg_pixels_affinity = fg_mask_affinity.numel() - fg_pixels_affinity
    # 计算前景权重（避免除零）
    fg_weight_affinity = bg_pixels_affinity / (fg_pixels_affinity + 1e-5) if fg_pixels_affinity > 0 else 1.0

    # 计算前景/背景像素数量（LSDS）
    fg_pixels_lsds = fg_mask_lsds.sum()
    bg_pixels_lsds = fg_mask_lsds.numel() - fg_pixels_lsds
    # 计算前景权重
    fg_weight_lsds = bg_pixels_lsds / (fg_pixels_lsds + 1e-5) if fg_pixels_lsds > 0 else 1.0

    # 生成像素级权重矩阵（前景区域用fg_weight，背景用1.0）
    weights_affinity = torch.where(
        torch.as_tensor(fg_mask_affinity == 1.0, device=gt_affinity.device),
        torch.as_tensor(fg_weight_affinity, device=gt_affinity.device),
        torch.as_tensor(1.0, device=gt_affinity.device)
    )
    weights_lsds = torch.where(
        torch.as_tensor(fg_mask_lsds == 1.0, device=gt_lsds.device),
        torch.as_tensor(fg_weight_lsds, device=gt_lsds.device),
        torch.as_tensor(1.0, device=gt_lsds.device)
    )

    # --------------------------
    # 2. 计算加权损失
    # --------------------------
    # 定义加权损失计算函数（以MSE为例，可根据实际loss_fn类型调整）
    def weighted_loss(pred, target, weights):
        # 计算基础损失（如MSE）
        base_loss = loss_fn(pred, target)
        # 应用权重（对每个元素的损失加权）
        weighted = base_loss * weights
        return weighted.mean()  # 加权后的平均损失

    # 计算加权损失（LSDS和亲和度分别加权）
    loss_lsd = weighted_loss(lsd_logits, gt_lsds, weights_lsds)
    loss_affinity = weighted_loss(affinity_logits, gt_affinity, weights_affinity)

    # 总损失（可根据任务重要性调整两者比例）
    loss_value = loss_lsd + loss_affinity  # 或 loss_lsd * 0.5 + loss_affinity * 0.5

    # 反向传播与参数更新
    if train_step:
        loss_value.backward()
        optimizer.step()

    # 激活输出
    lsd_output = activation(lsd_logits)
    affinity_output = activation(affinity_logits)

    outputs = {
        'pred_lsds': lsd_output,
        'lsds_logits': lsd_logits,
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }
    return loss_value, outputs


def trainer(num_fmaps: int, fmap_inc_factors: int, downsample_times: int, weighted=False, auto_loss=False):
    model = ACRLSD_2D(num_fmaps, fmap_inc_factors, downsample_times)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])

    # num_fmaps | fmap_inc_factors | downsample_factors
    Save_Name = (f'ACRLSD_2D(ninanjie)_all/{crop_size}_{num_fmaps}_{fmap_inc_factors}_{downsample_times}/'
                 f'{"auto" if auto_loss else "normal"}_{"weighted" if weighted else "unweighted"}')

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    output_dir = f'./output/log/{Save_Name}'
    os.makedirs(output_dir, exist_ok=True)
    logfile = f'{output_dir}/log.txt'
    fh = logging.FileHandler(logfile, mode='a', delay=False)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    writer = SummaryWriter(log_dir=output_dir)

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
    # set loss function
    loss_fn = torch.nn.MSELoss().to(device)

    # training loop
    model.train()
    loss_fn.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 50
    no_improve_count = 0

    model_step_fn = weighted_model_step if weighted else model_step

    with tqdm(total=training_epochs) as pbar:
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            epoch += 1
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in train_loader:
                ##Get Tensor
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                              device=device)  # (batch, 2, height, width)
                count += 1
                progress_desc = f"Train {count}/{total}, "
                train_desc = f"loss: {Best_val_loss:.4f}, "
                model_step_fn(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation)
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            acc_loss = []
            count, total = 0, len(val_loader)
            voi_affinity = 0.
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in val_loader:
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.uint8,
                                              device=device)  # (batch, 2, height, width)
                with torch.no_grad():
                    loss_value, outputs = model_step_fn(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation,
                                               train_step=False)
                # pred_lsds = outputs['pred_lsds']
                pred_affinity = outputs['pred_affinity']
                voi_affinity += np.sum(
                    variation_of_information(np.asarray(gt_affinity.detach().cpu().numpy().flatten() * 255.0, dtype=np.uint8),
                                             np.asarray(pred_affinity.detach().cpu().numpy().flatten() * 255.0, dtype=np.uint8))
                )
                count += 1
                acc_loss.append(loss_value)
                progress_desc = f"Val {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()
            voi_affinity /= total
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/voi_affinity', voi_affinity, epoch)

            random.seed(epoch)
            sample_idx = random.randint(0, pred_affinity.shape[0] - 1)
            pred_affinity = np.asarray(pred_affinity[sample_idx, ...].cpu().numpy() * 255.0, dtype=np.uint8) # (2, H, W)
            pred_affinity_unify = normalize_affinity(pred_affinity) # (2, H, W)
            gt_affinity = normalize_affinity(gt_affinity[sample_idx, ...].cpu().numpy() * 255.0) # (2, H, W)
            visual_pred_affinity = np.sum(pred_affinity, axis=0) # (H, W)
            visual_pred_affinity_unify = np.sum(pred_affinity_unify, axis=0)
            visual_gt_affinity = np.sum(gt_affinity, axis=0)
            # visual_raw = raw[batch_size >> 2, 0, ...].cpu().numpy()

            visualize_and_save_affinity(visual_pred_affinity, epoch, 'visual/pred', writer)
            visualize_and_save_affinity(visual_pred_affinity_unify, epoch, 'visual/pred_unify', writer)
            visualize_and_save_affinity(visual_gt_affinity, epoch, 'visual/gt', writer)
            # visualize_and_save_affinity(visual_raw, epoch, 'visual/raw', writer)
            writer.flush()

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                Best_model = model.state_dict()
                torch.save(Best_model, '{}/Best_in_val.model'.format(output_dir))
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = f'best epoch {Best_epoch}'
            ## Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))

            pbar.update(1)

            ##Early stop
            if no_improve_count > early_stop_count and epoch > 100:
                logging.info("Early stop!")
                break
            fh.flush()
    del model, Best_model, optimizer, activation, loss_fn
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 20

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,5,0,3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    ##装载数据
    crop_size = 512
    dataset_names = ['best_val_3']
    crop_xyz = [7, 7, 1]
    train_dataset, val_dataset = [], []
    for dataset_name, i, j, k in itertools.product(
            dataset_names,
            range(crop_xyz[0]),
            range(crop_xyz[1]),
            range(crop_xyz[2])
    ):
        train_tmp, val_tmp = load_train_dataset(dataset_name, raw_dir='raw_2',
                                                label_dir='truth_label_2_seg_1',
                                                from_temp=True, require_xz_yz=False, crop_size=crop_size,
                                                crop_xyz=crop_xyz, chunk_position=[i, j, k])
        train_dataset.append(train_tmp)
        val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=28,
                            pin_memory=True, collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader):
        res = []
        for raw, labels, point_map, mask, gt_affinity, gt_lsds in loader:
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append((raw, labels, point_map, mask, gt_affinity, gt_lsds))
        return res

    # train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)

    trainer(num_fmaps=32, fmap_inc_factors=5, downsample_times=3, weighted=True, auto_loss=True)
    # trainer(num_fmaps=32, fmap_inc_factors=5, downsample_times=3, weighted=False, auto_loss=True)
    # trainer()
