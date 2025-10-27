import copy
import gc
import itertools
import logging
import multiprocessing
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.metrics import variation_of_information
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DistributedDataParallel

# from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3d
from training.models.ACRLSD import ACRLSD_3D
from training.models.losses import WeightedMSELoss, MultiLosses
from training.utils.aftercare import visualize_and_save_affinity, normalize_affinity
from training.utils.dataloader_ninanjie import collate_fn_3D_ninanjie_Train_affinity, Dataset_3D_ninanjie_Train_GPU
from utils.dataloader_ninanjie import load_train_dataset, collate_fn_3D_ninanjie_Train


# from utils.dataloader_cremi import Dataset_3D_cremi_Train,collate_fn_3D_cremi_Train

## CUDA_VISIBLE_DEVICES=1 python main_ACRLSD_3d_train_zebrafinch.py &

def set_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


def model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, scaler=None, train_step=True, weight_affinity=False):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    with torch.cuda.amp.autocast():
        lsd_logits, affinity_logits = model(raw)
        weight = torch.ones_like(raw)
        if weight_affinity:
            pseudo_mask = (gt_affinity.sum(dim=1, keepdim=True) > 0) + 0 # (B, 1, H, W, N)
            bg = torch.sum(pseudo_mask)
            fg = torch.numel(pseudo_mask) - bg
            weight = torch.where(gt_affinity.sum(dim=1, keepdim=True) > 0, bg / fg, 1.0)  # (B, 1, H, W, N)
        loss_value = loss_fn(lsd_logits, gt_lsds, weight) + loss_fn(affinity_logits, gt_affinity, weight)

    # backward if training mode
    if train_step:
        scaler.scale(loss_value).backward()  # 替代 loss_value.backward()
        scaler.step(optimizer)  # 替代 optimizer.step()
        scaler.update()

    lsd_output = activation(lsd_logits)
    affinity_output = activation(affinity_logits)

    outputs = {
        'pred_lsds': lsd_output,
        'lsds_logits': lsd_logits,
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }

    return loss_value, outputs


def weighted_model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    # forward
    lsd_logits, affinity_logits = model(raw)

    loss_value = loss_fn(lsd_logits, gt_lsds) + loss_fn(affinity_logits, gt_affinity)

    # backward if training mode
    if train_step:
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


def write_log(writer, epoch, val_loss, voi_val, raw, gt_affinity, pred_affinity):
    writer.add_scalar('val_loss', val_loss, epoch)
    writer.add_scalar('val_voi', voi_val, epoch)

    # 可视化
    sample_idx = random.randint(0, raw.shape[0] - 1)
    if pred_affinity is not None:
        raw_np = raw[sample_idx, ..., -1].squeeze()  # (B, 1, H, W, N) -> (H, W)
        pred_affinity_np = np.asarray(pred_affinity[sample_idx, :, :, :, -1] * 255.0,
                                      dtype=np.uint8)  # (B, 3, H, W, N) -> (3, H, W)
        gt_affinity_np = gt_affinity[sample_idx, :, :, :, -1] * 255.0

        visual_pred_affinities = [normalize_affinity(np.sum(pred_affinity_np[layer, ...], axis=0))
                                  for layer in ([0, 1], [0, 2], [1, 2])]
        visual_gt_affinities = [normalize_affinity(np.sum(gt_affinity_np[layer, ...], axis=0))
                                for layer in ([0, 1], [0, 2], [1, 2])]

        visualize_and_save_affinity(raw_np, epoch, 'visual/raw', writer)
        visualize_and_save_affinity(visual_pred_affinities[0], epoch, 'visual/pred_xy', writer)
        visualize_and_save_affinity(visual_gt_affinities[0], epoch, 'visual/gt_xy', writer)
        visualize_and_save_affinity(visual_pred_affinities[1], epoch, 'visual/pred_xz', writer)
        visualize_and_save_affinity(visual_gt_affinities[1], epoch, 'visual/gt_xz', writer)
        visualize_and_save_affinity(visual_pred_affinities[2], epoch, 'visual/pred_yz', writer)
        visualize_and_save_affinity(visual_gt_affinities[2], epoch, 'visual/gt_yz', writer)

        writer.flush()


def trainer(num_fmaps: int, fmap_inc_factors: int, weight_affinity=False, auto_loss=False):
    model = ACRLSD_3D(num_fmaps=num_fmaps, fmap_inc_factor=fmap_inc_factors)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    # model = DistributedDataParallel(
    #     model.cuda(),
    #     device_ids=[local_rank],  # 当前进程使用的显卡（0-5）
    #     output_device=0,  # 所有输出聚合到0号卡
    #     # find_unused_parameters=True,  # 允许模型部分参数不参与计算
    #     broadcast_buffers=False
    # )
    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = load_data_single()
    # return

    ##创建log日志
    logger = None
    writer = None
    if local_rank == 0:
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        log_dir = f'./output/log/{Save_Name}/{crop_size}_{num_fmaps}_{fmap_inc_factors}_cpu'
        if weight_affinity:
            log_dir += '_weight_affinity'
        if auto_loss:
            log_dir += '_auto_loss'
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            return
        logfile = '{}/log.txt'.format(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        logging.info(f'''Starting training:
            training_epochs:  {training_epochs}
            Num_slices_train: {len(train_loader) * batch_size}
            Num_slices_val:   {len(val_loader) * batch_size}
            Batch size:       {batch_size}
            Learning rate:    {learning_rate}
            ''')

    ##开始训练
    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    activation = torch.nn.Sigmoid().to(device)
    loss_fn = MultiLosses([
        WeightedMSELoss(),
    ], auto_loss=auto_loss).to(device)

    model.train()
    loss_fn.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 35
    no_improve_count = 0
    pbar = tqdm(total=training_epochs) if local_rank == 0 else None
    progress_desc, train_desc, val_desc = '', '', ''
    log_proc = None

    while epoch < training_epochs:
        ###################Train###################
        model.train()
        # reset data loader to get random augmentations
        np.random.seed()
        random.seed()
        tmp_loader = iter(train_loader)
        count, total = 0, len(tmp_loader)
        for raw, gt_affinity, gt_lsds in tmp_loader:
            raw = torch.as_tensor(raw, device=device)
            gt_affinity = torch.as_tensor(gt_affinity, device=device)
            gt_lsds = torch.as_tensor(gt_lsds, device=device)
            model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, scaler)
            # torch.cuda.empty_cache()

            count += 1
            if local_rank == 0 and count & 7 == 0:
                progress_desc = f"Train {count}/{total}, "
                train_desc = f"loss: {Best_val_loss:.4f}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

        gc.collect()

        ###################Validate###################
        model.eval()
        ##Fix validation set
        seed = 98
        np.random.seed(seed)
        random.seed(seed)
        tmp_val_loader = iter(val_loader)
        acc_loss = []
        voi_val = 0
        count, total = 0, len(tmp_val_loader)
        for raw, gt_affinity, gt_lsds in tmp_val_loader:
            with torch.no_grad():
                raw = torch.as_tensor(raw, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, device=device)
                gt_lsds = torch.as_tensor(gt_lsds, device=device)
                loss_value, outputs = model_step(model, loss_fn, optimizer, raw, gt_lsds,
                                                 gt_affinity,
                                                 activation, train_step=False)

                # 收集所有进程的损失值进行平均
                # loss_value_tensor = torch.tensor([loss_value.item()], device=device)
                # dist.all_reduce(loss_value_tensor, op=dist.ReduceOp.SUM)
                # avg_loss = loss_value_tensor.item() / dist.get_world_size()
                # torch.cuda.empty_cache()

                y_affinity = outputs['pred_affinity']
                binary_gt_affinity = np.asarray(gt_affinity.cpu(), dtype=np.uint8).flatten()
                binary_y_affinity = np.asarray(y_affinity.cpu(), dtype=np.uint8).flatten()
                voi_val += np.sum(variation_of_information(binary_gt_affinity, binary_y_affinity))

                # 保存最后一个批次的预测结果用于可视化
                if count == total - 1:
                    pred_affinity = outputs['pred_affinity']

                count += 1
                acc_loss.append(loss_value.item())

                if count & 7 == 0:
                    progress_desc = f"Val {count}/{total}, "
                    pbar.set_description(progress_desc + train_desc + val_desc)

        # 计算平均验证损失
        val_loss = np.mean(acc_loss)

        voi_val /= (total + 0.0)

        if Best_val_loss > val_loss:
            Best_val_loss = val_loss
            Best_epoch = epoch
            torch.save(model.module.state_dict(), f'{log_dir}/Best_in_val.model')
            no_improve_count = 0
        else:
            no_improve_count += 1

        val_desc = f'VOI: {voi_val:.4f}, best epoch {Best_epoch}'
        pbar.update(1)

        # 记录日志
        logging.info(
            f"Epoch {epoch}: val_loss = {val_loss:.6f}, with best val_loss = {Best_val_loss:.6f} in epoch {Best_epoch}")

        write_log(writer, epoch, val_loss, voi_val,
                  raw.cpu().numpy(), gt_affinity.cpu().numpy(), pred_affinity.cpu().numpy())
        # 早停检查
        if no_improve_count == early_stop_count and epoch > 100:
            logging.info("Early stop!")
            torch.save(model.module.state_dict(), f'{log_dir}/final.model')
            writer.close()
            break

        torch.cuda.empty_cache()
        gc.collect()

        epoch += 1
    pbar.close()


def _load_datasets(dataset_name_, crop_size_, crop_xyz_, a, b, c):
    return load_train_dataset(dataset_name_, raw_dir='raw_2', label_dir='truth_label_2_seg_1', from_temp=True,
                              require_xz_yz=False, require_lsd=True,
                              crop_size=crop_size_, crop_xyz=crop_xyz_, chunk_position=[a, b, c])

def load_data():
    dataset_names = ['best_val_3_3d_cpu']
    crop_size = 384
    crop_xyz = [9, 9, 1]
    train_dataset, val_dataset = [], []

    # 只在主进程加载数据，其他进程等待
    if local_rank == 0:
        for dataset_name, i, j, k in itertools.product(
                dataset_names,
                range(crop_xyz[0]),
                range(crop_xyz[1]),
                range(crop_xyz[2])
        ):
            train_, val_ = _load_datasets(dataset_name, crop_size, crop_xyz, i, j, k)
            train_dataset.append(train_)
            val_dataset.append(val_)

        random.shuffle(train_dataset)
        train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)

    # 广播数据集到所有进程
    train_dataset = dist.broadcast_object_list([train_dataset], src=0)[0]
    val_dataset = dist.broadcast_object_list([val_dataset], src=0)[0]

    # 使用DistributedSampler进行数据分片
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # 数据加载器：关闭DataLoader的shuffle（由sampler负责）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # 必须用sampler，不能用shuffle=True
        num_workers=40,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn_3D_ninanjie_Train_affinity
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=40,
        pin_memory=False,
        collate_fn=collate_fn_3D_ninanjie_Train_affinity
    )

    return train_loader, val_loader


def load_data_single():
    train_dataset, val_dataset = [], []

    with multiprocessing.Pool(36) as pool:
        results = []
        for dataset_name, i, j, k in itertools.product(
                dataset_names,
                range(2, crop_xyz[0] - 2),
                range(2, crop_xyz[1] - 2),
                range(crop_xyz[2])
        ):
            results.append(pool.apply_async(_load_datasets, args=(dataset_name, crop_size, crop_xyz, i, j, k)))
        pool.close()
        pool.join()
        for res in results:
            train_, val_ = res.get()
            train_dataset.append(train_)
            val_dataset.append(val_)

    # for dataset_name, i, j, k in itertools.product(
    #         dataset_names,
    #         range(1, crop_xyz[0]-1),
    #         range(1, crop_xyz[1]-1),
    #         range(crop_xyz[2])
    # ):
    #     train_, val_ = _load_datasets(dataset_name, crop_size, crop_xyz, i, j, k)
    #     train_dataset.append(train_)
    #     val_dataset.append(val_)
    random.shuffle(val_dataset)
    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_3D_ninanjie_Train_affinity)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=14, pin_memory=True,
                            collate_fn=collate_fn_3D_ninanjie_Train_affinity)

    return train_loader, val_loader


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 20

    set_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1,5,0,2,3'
    local_rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    dataset_names = ['best_val_3_3d_cpu']
    crop_size = 256
    crop_xyz = [15, 15, 1]

    Save_Name = f'ACRLSD_3D(ninanjie)_all'

    # trainer(num_fmaps=34, fmap_inc_factors=4, weight_affinity=True)
    trainer(num_fmaps=30, fmap_inc_factors=5)
    trainer(num_fmaps=30, fmap_inc_factors=5, weight_affinity=True)

# if __name__ == '__main__':
#     set_seed()
#     Save_Name = f'ACRLSD_3D(ninanjie)_all'
#
#     # 显式设置环境变量（关键）
#     os.environ['CUDA_VISIBLE_DEVICES'] = "2,1,0"  # 使用所有6张卡
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29501'
#     gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=gpu_num,  # 总进程数=显卡数=6
#         rank=int(os.environ['RANK'])
#     )
#
#     # 2. 每个进程绑定到不同显卡（0-5号卡）
#     local_rank = int(os.environ['LOCAL_RANK'])
#     torch.cuda.set_device(local_rank)
#     device = torch.device('cuda', local_rank)
#
#     # 3. 超参数设置
#     num_fmaps = 21
#     fmap_inc_factors = 5
#     batch_size = 1  # 单进程batch size，总batch size要乘显卡数
#     learning_rate = 1e-4
#     training_epochs = 10000
#
#     trainer(num_fmaps, fmap_inc_factors)
