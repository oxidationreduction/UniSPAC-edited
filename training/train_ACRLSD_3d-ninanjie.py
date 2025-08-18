import math

import joblib
import numpy as np
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
from tqdm.auto import tqdm
import logging
# from torch.utils.tensorboard import SummaryWriter
from models.unet3d import UNet3d
from utils.dataloader_ninanjie import load_train_dataset, collate_fn_3D_ninanjie_Train


# from utils.dataloader_cremi import Dataset_3D_cremi_Train,collate_fn_3D_cremi_Train

## CUDA_VISIBLE_DEVICES=1 python main_ACRLSD_3d_train_zebrafinch.py &

def set_seed(seed = 1998):
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
            in_channels=1, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True).to(device)
        
        self.lsd_predict = torch.nn.Conv3d(in_channels=12,out_channels=10, kernel_size=1)  #最终输出层的卷积操作


        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True).to(device)
        
        self.affinity_predict = torch.nn.Conv3d(in_channels=12,out_channels=3, kernel_size=1)  #最终输出层的卷积操作
    
    def forward(self, x):
        
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x,y_lsds.detach()],dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds,y_affinity


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


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 1
    # Save_Name = 'ACRLSD_3D(hemi+fib25+cremi)'
    # Save_Name = 'ACRLSD_3D(ninanjie)'

    set_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,1,3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    model = ACRLSD_3d().to(device)
    model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    Save_Name = 'ACRLSD_3D(ninanjie)'

    dataset_names = ['best_truth_3d']
    crop_xyz = [4, 3, 1]
    train_dataset, val_dataset = [], []
    for dataset_name in dataset_names:
        for i in range(crop_xyz[0]-1):
            for j in range(crop_xyz[1]):
                for k in range(crop_xyz[2]):
                    _train, _val = load_train_dataset(dataset_name, raw_dir='raw_2', label_dir='truth_label_2', from_temp=True,
                                                      crop_size=320, crop_xyz=crop_xyz, chunk_position=[i, j, k])
                    train_dataset.append(_train)
                    val_dataset.append(_val)
    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True,
                              drop_last=False,collate_fn=collate_fn_3D_ninanjie_Train)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False, num_workers=28, pin_memory=True,
                            collate_fn=collate_fn_3D_ninanjie_Train)

    def load_data_to_device(loader):
        res = []
        for raw, labels, mask_3D, gt_affinity, point_map, gt_lsds in loader:
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            # labels = torch.as_tensor(labels, dtype=torch.int32, device=device)
            # mask_3D = torch.as_tensor(mask_3D, dtype=torch.uint8, device=device)
            # point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
            gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append((raw, labels, mask_3D, gt_affinity, point_map, gt_lsds))
        return res

    train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)

    ##创建log日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = './output/log/log_{}.txt'.format(Save_Name)
    fh = logging.FileHandler(logfile, mode='a', delay=False)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

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
    early_stop_count = 100
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
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
            count, total = 0, len(tmp_val_loader)
            for raw, labels, mask_3D, gt_affinity, point_map, gt_lsds in tmp_val_loader:
                with torch.no_grad():
                    loss_value, _ = model_step(model, loss_fn, optimizer, raw, gt_lsds, gt_affinity, activation, train_step=False)
                count += 1
                acc_loss.append(loss_value)
                progress_desc = f"Val {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(),'./output/checkpoints/{}_Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = f'best epoch {Best_epoch}'
            pbar.update(1)

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch,val_loss,Best_val_loss,Best_epoch))
            # writer.add_scalar('val_loss', val_loss, epoch)
        
            ##Early stop
            if no_improve_count==early_stop_count and epoch > 100:
                logging.info("Early stop!")
                break
