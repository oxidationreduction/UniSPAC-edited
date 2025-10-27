import gc
import itertools
import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from skimage.metrics import variation_of_information
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# from torch.utils.tensorboard import SummaryWriter
from models.unet2d import UNet2d
from training.models.ACRLSD import ACRLSD_2D, remove_module
from training.models.losses import DiceLoss, WeightedDiceLoss
from training.models.segEM import segEM2d
from training.utils.aftercare import aftercare, visualize_and_save_mask
from training.utils.dataloader_ninanjie import get_acc_prec_recall_f1
from utils.dataloader_ninanjie import load_train_dataset, collate_fn_2D_fib25_Train

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


def model_step(model, optimizer, input_prompt, y_affinity_logits, gt_binary_mask, gt_affinity,
               activation, scalar=None, train_step=True, auto_loss=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    with ((torch.cuda.amp.autocast(enabled=scalar is not None))):
        y_mask_logits = model(input_prompt, y_affinity_logits)
        y_mask_ = y_mask_logits.squeeze()
        gt_binary_mask_ = gt_binary_mask.squeeze()
        loss1 = F.binary_cross_entropy_with_logits(y_mask_, gt_binary_mask_)

        y_mask = activation(y_mask_logits)
        Diceloss_fn = DiceLoss().to(device)
        loss2 = Diceloss_fn(1 - y_mask, 1 - gt_binary_mask)

        loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)

        if auto_loss:
            loss = loss1 / (loss1.detach().cpu() + 1e-6)
            + loss2 / (loss2.detach().cpu() + 1e-6)
            + loss3 / (loss3.detach().cpu() + 1e-6)
        else:
            loss = loss1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

    # backward if training mode
    if train_step:
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        y_mask = (y_mask_ > 0.5) + 0.

    return None if train_step else (y_mask, [loss1, loss2, loss3])

# point_map, mask, y_affinity_logits, gt_affinity,
def weighted_model_step(model, optimizer, input_prompt, y_affinity_logits, gt_binary_mask, gt_affinity,
                        activation, scalar=None, train_step=True, auto_loss=True):
    # zero gradients if training
    if train_step:
        optimizer.zero_grad()

    with ((torch.cuda.amp.autocast(enabled=scalar is not None))):
        y_mask_logits = model(input_prompt, y_affinity_logits)
        y_mask_ = y_mask_logits.squeeze()
        gt_binary_mask_ = gt_binary_mask.squeeze()

        foreground = gt_binary_mask_.sum()
        total = gt_binary_mask_.numel()
        background = total - foreground
        weight = background / foreground

        weights = torch.where(torch.as_tensor(gt_binary_mask_ == 1), weight, torch.tensor(1.0, device=device))
        loss1 = F.binary_cross_entropy_with_logits(y_mask_, gt_binary_mask_, weight=weights)

        y_mask = activation(y_mask_logits)
        Diceloss_fn = WeightedDiceLoss().to(device)
        loss2 = Diceloss_fn(1. - y_mask, 1. - gt_binary_mask, weights)

        loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)

        if auto_loss:
            loss = loss1 / (loss1.detach().cpu() + 1e-6) \
                   + loss2 / (loss2.detach().cpu() + 1e-6) \
                   + loss3 / (loss3.detach().cpu() + 1e-6)
        else:
            loss = loss1 + loss2 * WEIGHT_LOSS2 + loss3 * WEIGHT_LOSS3

    # backward if training mode
    if train_step:
        if scalar is not None:
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        y_mask, gt_binary_mask = (y_mask_ > 0.5) + 0., gt_binary_mask_

    return None if train_step else (y_mask, (loss1, loss2, loss3))


def trainer(num_fmaps=24, fmap_inc_factors=4, downsample_times=4, weighted=False, auto_loss=True):
    model = segEM2d(
        num_fmaps=num_fmaps, fmap_inc_factors=fmap_inc_factors, downsample_times=downsample_times
    )
    model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device)
    scalar = torch.cuda.amp.GradScaler()

    ## 创建log日志
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    Save_Name = '{}_{}_{}_{}'.format(crop_size, num_fmaps, fmap_inc_factors, downsample_times)
    Save_Name = ('auto-' if auto_loss else f'w2-{WEIGHT_LOSS2}-w3-{WEIGHT_LOSS3}-') + Save_Name
    Save_Name = ('weighted-' if weighted else 'unweighted-')+ Save_Name
    model_step_fn = weighted_model_step if weighted else model_step
    log_dir = f'./output/log/segEM2d(ninanjie)-prompt-3rd/{Save_Name}'
    os.makedirs(log_dir, exist_ok=True)
    for filename in os.listdir(log_dir):
        file_path = os.path.join(log_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    logfile = os.path.join(log_dir, 'log.txt'.format(Save_Name))
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

    writer = SummaryWriter(log_dir=log_dir)

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
    early_stop_count = 50
    no_improve_count = 0
    with (tqdm(total=training_epochs) as pbar):
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            train_desc = f"Best: {Best_epoch}, loss: {Best_val_loss:.2f}, "
            for raw, labels, point_map, mask, gt_affinity, y_affinity_logits in train_loader:
                model_step_fn(model, optimizer, point_map, y_affinity_logits, mask, gt_affinity, activation,
                              scalar=scalar, train_step=True, auto_loss=auto_loss)
                count += 1
                progress_desc = f'Train: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)
            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            acc_loss = []
            detailed_losses = np.array([0.] * 3)
            count, total = 0, len(val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            metrics = np.asarray((0, 0, 0, 0), dtype=np.float64)
            metrics_aftercare = np.asarray((0, 0, 0, 0), dtype=np.float64)
            voi_val, voi_val_aftercare = 0, 0
            for raw, labels, point_map, mask, gt_affinity, y_affinity_logits in val_loader:
                with torch.no_grad():
                    y_mask, losses = model_step_fn(model, optimizer, point_map, y_affinity_logits, mask, gt_affinity,
                                                   activation, scalar=scalar, train_step=False, auto_loss=auto_loss)
                    loss_value = sum(losses)
                    acc_loss.append(loss_value)
                    binary_y_mask = np.asarray(y_mask.detach().cpu(), dtype=np.uint8).flatten()
                    binary_y_mask_aftercare = aftercare(y_mask.detach().unsqueeze(0).unsqueeze(-1).cpu()).squeeze().flatten()
                    binary_gt_seg = ((np.asarray(labels) > 0.0) + 0).flatten()

                    metrics += np.asarray(get_acc_prec_recall_f1(binary_y_mask, binary_gt_seg))
                    metrics_aftercare += np.asarray(get_acc_prec_recall_f1(binary_y_mask_aftercare, binary_gt_seg))
                    voi_val += np.sum(variation_of_information(binary_y_mask, binary_gt_seg))
                    voi_val_aftercare += np.sum(variation_of_information(binary_y_mask_aftercare, binary_gt_seg))
                    detailed_losses += np.asarray([loss_.cpu().detach().numpy() for loss_ in losses])

                count += 1
                progress_desc = f'Val: {count}/{total}, '
                pbar.set_description(progress_desc + train_desc + val_desc)

            metrics /= (total + 0.0)
            voi_val /= (total + 0.0)
            detailed_losses /= (total + 0.)
            bce_, dice_, affinity_ = detailed_losses[:]
            acc, prec, recall, f1 = metrics[:]
            val_desc = f'acc: {acc:.3f}, prec: {prec:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, VOI: {voi_val:.5f}'
            # val_loss = np.mean(np.array([loss_value.cpu().numpy() for loss_value in acc_loss]))
            val_loss = torch.stack([loss_value.cpu() for loss_value in acc_loss]).mean().item()

            y_mask = y_mask.squeeze().detach().cpu()
            random.seed(epoch)
            sample_idx = random.randint(0, y_mask.shape[0] - 1)
            y_mask = torch.stack([torch.as_tensor(measure.label(y_mask[idx, :, :])) for idx in range(y_mask.shape[0])], dim=0)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), y_mask[sample_idx, :, :], idx=epoch, mode='visual/seg_only', writer=writer)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), y_mask[sample_idx, :, :], idx=epoch, mode='visual/raw_seg', writer=writer)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), labels[sample_idx, :, :], idx=epoch, mode='visual/gt', writer=writer)
            y_mask = aftercare(y_mask.detach().unsqueeze(0).unsqueeze(-1).cpu()).squeeze()  # (B, H, W)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), y_mask[sample_idx, :, :], idx=epoch, mode='aftercare/seg_only', writer=writer)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), y_mask[sample_idx, :, :], idx=epoch, mode='aftercare/raw_seg', writer=writer)
            visualize_and_save_mask(raw.squeeze()[[sample_idx], :, :].cpu(), labels[sample_idx, :, :], idx=epoch, mode='aftercare/gt', writer=writer)
            writer.add_scalar('aftercare/VOI', voi_val_aftercare, epoch)
            writer.add_scalar('aftercare/acc', metrics_aftercare[0] / (total + 0.0), epoch)
            writer.add_scalar('aftercare/prec', metrics_aftercare[1] / (total + 0.0), epoch)
            writer.add_scalar('aftercare/recall', metrics_aftercare[2] / (total + 0.0), epoch)
            writer.add_scalar('aftercare/f1', metrics_aftercare[3] / (total + 0.0), epoch)

            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('loss/bce', bce_, epoch)
            writer.add_scalar('loss/dice', dice_, epoch)
            writer.add_scalar('loss/affinity', affinity_, epoch)
            writer.add_scalar('VOI', voi_val, epoch)
            writer.add_scalar('metrics/acc', acc, epoch)
            writer.add_scalar('metrics/prec', prec, epoch)
            writer.add_scalar('metrics/recall', recall, epoch)
            writer.add_scalar('metrics/f1', f1, epoch)
            writer.flush()

            ###################Compare###################
            if Best_val_loss > val_loss:
                Best_val_loss = val_loss
                Best_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(log_dir, 'Best_in_val.model'.format(Save_Name)))
                no_improve_count = 0
            else:
                no_improve_count += 1

            ##Record
            logging.info("Epoch {}: val_loss = {:.6f},with best val_loss = {:.6f} in epoch {}".format(
                epoch, val_loss, Best_val_loss, Best_epoch))
            fh.flush()
            # writer.add_scalar('val_loss', val_loss, epoch)

            epoch += 1
            pbar.update(1)
            ##Early stop
            if no_improve_count == early_stop_count:
                logging.info("Early stop!")
                break

    del model, optimizer, activation
    gc.collect()

def get_affinity_model():
    model_affinity = ACRLSD_2D(num_fmaps=32, fmap_inc_factors=5, downsample_times=3)
    model_path = ('/home/liuhongyu2024/Documents/UniSPAC-edited/training/output/log/'
                  'ACRLSD_2D(ninanjie)_all/256_32_5_3/auto_unweighted/Best_in_val.model')
    weights = torch.load(model_path, map_location=torch.device('cuda'))
    model_affinity.load_state_dict(remove_module(weights))
    for param in model_affinity.parameters():
        param.requires_grad = False
    return model_affinity.cuda()


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 1000
    learning_rate = 1e-4
    batch_size = 40

    set_seed()
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,5,0,3'  # 设置所有可以使用的显卡，共计四块,
    device_ids = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]   # 选中显卡
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model_affinity = get_affinity_model()

    ##装载数据
    dataset_names = ['best_val_3_cpu']

    ##装载数据
    crop_size = 256
    crop_xyz = [15, 15, 1]
    train_dataset, val_dataset = [], []
    for dataset_name in dataset_names:
        for i, j, k in itertools.product(
                range(crop_xyz[0]),
                range(crop_xyz[1]),
                range(crop_xyz[2])
        ):
            train_tmp, val_tmp = load_train_dataset(dataset_name, raw_dir='raw_2', label_dir='truth_label_2_seg_1',
                                                    from_temp=True, require_xz_yz=False, crop_size=crop_size,
                                                    crop_xyz=crop_xyz, chunk_position=[i, j, k])
            train_dataset.append(train_tmp)
            val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True,
                              drop_last=False, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=28, pin_memory=True,
                            collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader):
        tmp_loader = iter(loader)
        res = []
        # 先将model_affinity设为eval模式（已在main中设置，此处可再次确认）
        model_affinity.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # 禁用autocast，用FloatTensor计算（可选）
            for raw, labels, point_map, mask, gt_affinity, _ in tqdm(tmp_loader, leave=True, desc='load to cuda'):
                ## Get Tensor并移到设备
                raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, H, W)
                point_map = torch.as_tensor(point_map, dtype=torch.float, device=device)
                mask = torch.as_tensor(mask, dtype=torch.float, device=device)
                gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float, device=device)
                # 提前计算y_affinity_logits并存储
                _, y_affinity_logits = model_affinity(raw)
                res.append([raw, labels, point_map, mask, gt_affinity, y_affinity_logits])
        return res

    train_loader, val_loader = load_data_to_device(train_loader), load_data_to_device(val_loader)
    del model_affinity
    torch.cuda.empty_cache()

    # trainer(32, 5, 3, weighted=False, auto_loss=True)
    trainer(32, 5, 3, weighted=False, auto_loss=True)
