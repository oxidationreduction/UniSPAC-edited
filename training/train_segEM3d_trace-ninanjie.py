import itertools
import math
from collections import OrderedDict

import joblib
import numpy as np
import os
import random
import torch
import torch.nn as nn
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.utils.data import DataLoader,ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import logging
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from models.unet3d import UNet3d
from training.utils.aftercare import visualize_and_save_mask
from utils.dataloader_hemi_better import Dataset_3D_hemi_Train,collate_fn_3D_hemi_Train
from utils.dataloader_ninanjie import load_train_dataset, \
    collate_fn_3D_ninanjie_Train, get_acc_prec_recall_f1


# from utils.dataloader_cremi import Dataset_3D_cremi_Train,collate_fn_3D_cremi_Train

## CUDA_VISIBLE_DEVICES=0 python main_segEM_3d_train_zebrafinch.py &

# WEIGHT_LOSS_AFFINITY = 10



def set_seed(seed = 19260817):
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
            constant_upsample=True)

        self.lsd_predict = torch.nn.Conv2d(in_channels=12, out_channels=6, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 6 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet2d(
            in_channels=7,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2], [2, 2], [2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.affinity_predict = torch.nn.Conv2d(in_channels=12, out_channels=2, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


####ACRLSD模型
class segEM_2d(torch.nn.Module):
    def __init__(
            self,
    ):
        super(segEM_2d, self).__init__()

        ##For affinity prediction
        self.model_affinity = ACRLSD()
        # model_path = './output/checkpoints/ACRLSD_2D(hemi+fib25+cremi)_Best_in_val.model'
        model_path = './output/checkpoints/ACRLSD_2D(ninanjie)_half_crop_Best_in_val.model'
        weights = torch.load(model_path, map_location=torch.device('cuda:0'))
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

        y_concat = torch.cat([x_prompt.unsqueeze(1), y_affinity], dim=1)

        y_mask = self.mask_predict(self.model_mask(y_concat))
        y_mask = self.sigmoid(y_mask)

        return y_mask, y_lsds, y_affinity


####ACRLSD_3d模型
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
            constant_upsample=True)

        self.lsd_predict = torch.nn.Conv3d(in_channels=12, out_channels=10, kernel_size=1)  # 最终输出层的卷积操作

        # create our network, 10 input channels in the lsds data and 1 input channels in the raw data
        self.model_affinity = UNet3d(
            in_channels=11,  # 输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],  # 降采样的因子
            padding='same',
            constant_upsample=True)

        self.affinity_predict = torch.nn.Conv3d(in_channels=12, out_channels=3, kernel_size=1)  # 最终输出层的卷积操作

    def forward(self, x):
        y_lsds = self.lsd_predict(self.model_lsds(x))

        y_concat = torch.cat([x, y_lsds.detach()], dim=1)

        y_affinity = self.affinity_predict(self.model_affinity(y_concat))

        return y_lsds, y_affinity


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # 去掉 'module.' 前缀
        new_state_dict[name] = v
    return new_state_dict


class segEM_3d_trace(torch.nn.Module):
    def __init__(
        self,
    ):
        super(segEM_3d_trace, self).__init__()
        
        ##For affinity prediction
        self.model_affinity = ACRLSD_3d()
        # model_path = './output/checkpoints/ACRLSD_3D(hemi+fib25+cremi)_Best_in_val.model' 
        model_path = './output/log/ACRLSD_3D(ninanjie)_384/ACRLSD_3D(ninanjie)_384_Best_in_val.model'
        weights = torch.load(model_path,map_location=torch.device('cuda'))
        self.model_affinity.load_state_dict(remove_module(weights))
        for param in self.model_affinity.parameters():
            param.requires_grad = False
            
        # create our network, 3 input channels in the affinity data and 1 input channels in the raw data
        self.model_mask = UNet3d(
            in_channels=4, #输入的图像通道数
            num_fmaps=12,
            fmap_inc_factors=5,
            downsample_factors=[[2,2,2],[2,2,2],[2,2,2]], #降采样的因子
            padding='same',
            constant_upsample=True).cuda()
        
        self.mask_predict = torch.nn.Conv3d(in_channels=12,out_channels=1, kernel_size=1)  #最终输出层的卷积操作
        
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_raw, gt_mask2d_slice0):
        '''
        x_raw: shape = (Batch * channel * dim_x * dim_y * dim_z)
        gt_mask2d_slice0: shape = (Batch * dim_x * dim_y)
        '''
        # ##Get mask for slice0
        # y_mask2d_slice0,_,_ = self.model_mask_2d(x_raw[:,:,:,:,0],x_prompt)
        
        ##Get affinity for raw
        y_lsds,y_affinity = self.model_affinity(x_raw)
        
        #replace raw slice0
        x_raw_new = deepcopy(x_raw)
        x_raw_new[:,0,:,:,0] = gt_mask2d_slice0
        y_concat = torch.cat([x_raw_new,y_affinity.detach()],dim=1)

        y_mask3d = self.mask_predict(self.model_mask(y_concat))
        y_mask3d = self.sigmoid(y_mask3d)

        return y_mask3d,y_affinity,y_lsds

    
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

def model_step(model, optimizer, input_image, input_prompt, gt_binary_mask, train_step=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    gt_binary_mask = torch.as_tensor(gt_binary_mask, dtype=torch.float32, device=gt_binary_mask.device)
    y_mask,_,_ = model(input_image, gt_binary_mask[:,:,:,0])

    loss1 = F.binary_cross_entropy(y_mask.squeeze(),gt_binary_mask.squeeze())
    Diceloss_fn = DiceLoss().to(device)
    loss2 = Diceloss_fn(1-y_mask.squeeze(), 1-gt_binary_mask.squeeze())

    loss = loss1 + loss2
    
    # backward if training mode
    if train_step:
        loss.backward()
        optimizer.step()
    
    return loss, y_mask


def trainer(log_dir_):
    model = segEM_3d_trace()
    model = model.to(device)

    # ###多卡训练
    # ###一机多卡设置
    model = nn.DataParallel(model, device_ids=device_ids)  # 并行使用

    ##创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    log_dir_ = f'{log_dir_}/{Save_Name}'
    os.makedirs(log_dir_)

    logfile = '{}/log.txt'.format(log_dir_)
    csvfile = '{}/log.csv'.format(log_dir_)
    writer = SummaryWriter(log_dir=log_dir_)
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info(f'''Starting training:
        training_epochs:  {training_epochs}
        Num_patch_train:  {len(train_dataset)}
        Num_patch_val:    {len(val_dataset)}
        Batch size:       {batch_size}
        Learning rate:    {learning_rate}
        Device:           {device.type}
        ''')

    ##开始训练
    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # set activation
    activation = torch.nn.Sigmoid()

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 50
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        analysis = pd.DataFrame(columns=['bce', 'dice', 'affinity', 'acc', 'prec', 'recall', 'f1', 'voi'])
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            train_desc = f"loss: {Best_val_loss:.4f} in {Best_epoch}, "
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            tmp_loader = iter(train_loader)
            count, total = 0, len(tmp_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, mask, affinity, point_map in tmp_loader:
                ##Get Tensor
                # raw = torch.as_tensor(raw,dtype=torch.float, device= device)
                # point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                # mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                # affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 2, height, width)
                model_step(model, optimizer, raw, point_map, mask, affinity, activation, train_step=True)

                count += 1
                progress_desc = f"Train {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            detailed_losses = np.array([0.] * 3)
            voi_val = 0.
            judgement_rates = np.array([0.] * 4)
            count, total = 0, len(tmp_val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, mask, affinity, point_map in tmp_val_loader:
                # raw = torch.as_tensor(raw,dtype=torch.float, device= device) #(batch, 1, height, width, depth)
                # point_map = torch.as_tensor(point_map, dtype=torch.float, device=device) #(batch, height, width, depth)
                # mask = torch.as_tensor(mask, dtype=torch.float, device=device) #(batch, 2, height, width, depth)
                # affinity = torch.as_tensor(affinity, dtype=torch.float, device=device) #(batch, 2, height, width, depth)

                with torch.no_grad():
                    loss_value, pred, losses = model_step(model, optimizer, raw, point_map, mask, affinity, activation,
                                                          train_step=False)

                    binary_y_pred = np.asarray(pred.cpu(), dtype=np.uint8).flatten()
                    binary_gt_mask = np.asarray(mask.cpu(), dtype=np.uint8).flatten()

                    voi_val += np.sum(variation_of_information(binary_y_pred, binary_gt_mask))
                    judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred, binary_gt_mask))
                    acc_loss.append(loss_value.cpu().detach().numpy())
                    detailed_losses += np.asarray([loss_.cpu().detach().numpy() for loss_ in losses])

                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            detailed_losses /= (total + 0.)
            bce_, dice_, affinity_ = detailed_losses[:]
            voi_val /= (total + 0.)
            judgement_rates /= (total + 0.)
            acc, prec, recall, f1 = judgement_rates[:]
            val_loss = torch.as_tensor([loss_value.item() for loss_value in acc_loss]).mean().item()
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'
            analysis.loc[len(analysis)] = [bce_, dice_, affinity_, acc, prec, recall, f1, voi_val]

            visualize_and_save_mask(raw, pred, mode='visual/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw, pred, mode='visual/normal', idx=epoch, writer=writer)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), '{}/Best_in_val.model'.format(log_dir_))
                no_improve_count = 0
            else:
                no_improve_count += 1

            pbar.update(1)
            epoch += 1
            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('loss/bce', bce_, epoch)
            writer.add_scalar('loss/dice', dice_, epoch)
            writer.add_scalar('loss/affinity', affinity_, epoch)
            writer.add_scalar('metrics/accuracy', acc, epoch)
            writer.add_scalar('metrics/precision', prec, epoch)
            writer.add_scalar('metrics/recall', recall, epoch)
            writer.add_scalar('metrics/f1_score', f1, epoch)
            writer.add_scalar('VOI', voi_val, epoch)
            analysis.to_csv(csvfile)

            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break

    torch.save(model.module.state_dict(), f'{log_dir_}/final.model')
    del model, optimizer, activation
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 1000
    learning_rate = 1e-4
    batch_size = 40
    # Save_Name = 'segEM3d(hemi+fib25+cremi)'
    # Save_Name = 'segEM3d(hemi+fib25)faster_wloss3({})'.format(WEIGHT_LOSS_AFFINITY)
    Save_Name = './output/log/segEM3d_trace'
    if os.path.exists(Save_Name):
        os.remove(Save_Name)
    os.makedirs(Save_Name)

    set_seed()

    ###创建模型
    # set device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,1,5,0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = segEM_3d_trace()
    # model = model.to(device)
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]
    model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device_ids[0])#并行使用

    dataset_names = ['best_truth_3d']
    crop_size = 384
    crop_xyz = [4, 3, 1]
    train_dataset, val_dataset = [], []

    for dataset_name, i, j, k in itertools.product(
            dataset_names,
            range(crop_xyz[0] - 1),
            range(crop_xyz[1]),
            range(crop_xyz[2])
    ):
        train_, val_ = load_train_dataset(dataset_name, raw_dir='raw_2', label_dir='truth_label_2', from_temp=True,
                                          require_lsd=False, require_xz_yz=False,
                                          crop_size=crop_size, crop_xyz=crop_xyz, chunk_position=[i, j, k])
        train_dataset.append(train_)
        val_dataset.append(val_)

    train_dataset, val_dataset = ConcatDataset(train_dataset), ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn_3D_ninanjie_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False,
                            collate_fn=collate_fn_3D_ninanjie_Train)

    ##创建log日志
    writer = SummaryWriter(log_dir=Save_Name)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logfile = '{}/log.txt'.format(Save_Name)
    fh = logging.FileHandler(logfile,mode='a',delay=False)
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
    Num_patch_train:  {len(train_dataset)}
    Num_patch_val:    {len(val_dataset)}
    Batch size:       {batch_size}
    Learning rate:    {learning_rate}
    Device:           {device.type}
    ''')


    ##开始训练
    # set optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    # set activation
    activation = torch.nn.Sigmoid()

    # training loop
    model.train()
    epoch = 0
    Best_val_loss = 100000
    Best_epoch = 0
    early_stop_count = 50
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            train_desc = f"loss: {Best_val_loss:.4f} in {Best_epoch}, "
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            tmp_loader = iter(train_loader)
            count, total = 0, len(tmp_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, mask, affinity, point_map in tmp_loader:
                model_step(model, optimizer, raw, point_map, mask, train_step=True)
                count += 1
                progress_desc = f"Train {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 19260817
            np.random.seed(seed)
            random.seed(seed)
            tmp_val_loader = iter(val_loader)
            acc_loss = []
            voi_val = 0.
            judgement_rates = np.array([0.] * 4)
            count, total = 0, len(tmp_val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, mask, affinity, point_map in tmp_val_loader:
                with torch.no_grad():
                    loss_value, pred = model_step(model, optimizer, raw, point_map, mask, train_step=False)
                    binary_y_pred = np.asarray(pred.cpu(), dtype=np.uint8).flatten()
                    binary_gt_mask = np.asarray(mask.cpu(), dtype=np.uint8).flatten()

                    voi_val += np.sum(variation_of_information(binary_y_pred, binary_gt_mask))
                    judgement_rates += np.asarray(get_acc_prec_recall_f1(binary_y_pred, binary_gt_mask))
                acc_loss.append(loss_value.cpu().detach().numpy())
                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            val_loss = np.mean(acc_loss)
            voi_val /= (total + 0.)
            judgement_rates /= (total + 0.)
            acc, prec, recall, f1 = judgement_rates[:]
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'

            raw, pred = raw.cpu(), pred.cpu()
            visualize_and_save_mask(raw, pred, mode='visual/seg_only', idx=epoch, writer=writer)
            visualize_and_save_mask(raw, pred, mode='visual/normal', idx=epoch, writer=writer)
            writer.add_scalar('metrics/accuracy', acc, epoch)
            writer.add_scalar('metrics/precision', prec, epoch)
            writer.add_scalar('metrics/recall', recall, epoch)
            writer.add_scalar('metrics/f1_score', f1, epoch)
            writer.add_scalar('loss/VOI', voi_val, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(),'{}/Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count = no_improve_count + 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch,val_loss,Best_val_loss,Best_epoch))
            # writer.add_scalar('val_loss', val_loss, epoch)

            ##Early stop
            if no_improve_count==early_stop_count:
                logging.info("Early stop!")
                break

            epoch += 1
            pbar.update(1)
