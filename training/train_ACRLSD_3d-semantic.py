import gc
import itertools
import logging
import multiprocessing
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage.metrics import variation_of_information
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3d
from training.utils.dataloader_ninanjie import load_semantic_dataset, load_3d_semantic_dataset
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


####ACRLSD_3d模型

####ACRLSD模型
class ACRLSD_3d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(ACRLSD_3d, self).__init__()
        # d_factors = [[2,2,2],[2,2,2],[2,2,2]]  #降采样的因子
        # in_channels=1 #输入的图像通道数
        # num_fmaps=12
        # fmap_inc_factor=5

        # create our network, 1 input channels in the raw data
        self.model_lsds = UNet3d(
            in_channels=1,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.lsd_predict = torch.nn.Conv3d(in_channels=12, out_channels=10, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True).to(device)

        self.affinity_predict = torch.nn.Conv3d(in_channels=12, out_channels=3, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


def model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, train_step=True):
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
        'pred_affinity': (affinity_output > 0.5) + 0,
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


def _load_datasets(dataset_name_, crop_size_, crop_xyz_, a, b, c):
    return load_3d_semantic_dataset(dataset_name_, raw_dir='raw', label_dir='export', from_temp=True,
                                 crop_size=crop_size_, crop_xyz=crop_xyz_, chunk_position=[a, b, c])


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 24
    # Save_Name = 'ACRLSD_3D(hemi+fib25+cremi)'
    # Save_Name = 'ACRLSD_3D(ninanjie)'

    set_seed()

    dataset_names = ['fourth_1_3d', 'second_6_3d']
    crop_size = 384
    crop_xyz = [4, 3, 1]
    train_dataset, val_dataset = [], []

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,1,3,0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    model = ACRLSD_3d().to(device)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    Save_Name = f'ACRLSD_3D(semantic)_{crop_size}'

    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(14) as pool:
        results = []
        for dataset_name in dataset_names:
            for i, j, k in itertools.product(
                    range(crop_xyz[0] - 1),
                    range(crop_xyz[1]),
                    range(crop_xyz[2])
            ):
                results.append(pool.apply_async(_load_datasets, args=(dataset_name, crop_size, crop_xyz, i, j, k)))
        for result_ in results:
            train_dataset.append(result_.get())

    random.shuffle(train_dataset)
    val_size = len(train_dataset) // 6
    val_dataset = train_dataset[:val_size]
    train_dataset = train_dataset[val_size:]
    # for dataset_name, i, j, k in itertools.product(
    #         dataset_names,
    #         range(crop_xyz[0] - 1),
    #         range(crop_xyz[1]),
    #         range(crop_xyz[2])
    # ):
    #     train_, val_ = _load_datasets(dataset_name, crop_size, crop_xyz, i, j, k)
    #     train_dataset.append(train_)
    #     val_dataset.append(val_)
    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn_3D_ninanjie_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn_3D_ninanjie_Train)

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    log_dir = f'./output/log/{Save_Name}'
    os.makedirs(log_dir, exist_ok=True)
    logfile = '{}/log_{}.txt'.format(log_dir, Save_Name)
    csvfile = '{}/log_{}.csv'.format(log_dir, Save_Name)
    writer = SummaryWriter(log_dir=log_dir)
    fh = logging.FileHandler(logfile, mode='a')
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
    with tqdm(total=training_epochs) as pbar:
        analysis = pd.DataFrame(columns=['loss', 'voi'])
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            tmp_loader = iter(train_loader)
            count, total = 0, len(tmp_loader)
            for raw, labels, mask_3D, gt_affinity, point_map, gt_lsds in tmp_loader:
                model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation)

                count += 1
                progress_desc = f"Train {count}/{total}, "
                train_desc = f"loss: {Best_val_loss:.4f}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

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
            for raw, labels, mask_3D, gt_affinity, point_map, gt_lsds in tmp_val_loader:
                with torch.no_grad():
                    # loss_value, outputs = model_step(model, loss_fn, optimizer, _raw, _gt_lsds, _gt_affinity,
                    #                                  activation, train_step=False)
                    loss_value, outputs = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity,
                                                     activation, train_step=False)

                binary_gt_affinity = np.asarray(gt_affinity.cpu(), dtype=np.uint8).flatten()
                binary_y_affinity = np.asarray(outputs['pred_affinity'].cpu(), dtype=np.uint8).flatten()
                voi_val += np.sum(variation_of_information(binary_gt_affinity, binary_y_affinity))
                count += 1
                acc_loss.append(loss_value)
                progress_desc = f"Val {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()
            voi_val /= (total + 0.0)
            analysis.loc[len(analysis)] = [val_loss, voi_val]
            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(), '{}/{}_Best_in_val.model'.format(log_dir, Save_Name))
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = f'VOI: {voi_val:.4f}, best epoch {Best_epoch}'
            pbar.update(1)
            epoch += 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('val_voi', voi_val, epoch)
            analysis.to_csv(csvfile)

            ##Early stop
            if no_improve_count == early_stop_count and epoch > 100:
                logging.info("Early stop!")
                torch.save(model.module.state_dict(), '{}/{}_final.model'.format(log_dir, Save_Name))
                writer.close()
                break