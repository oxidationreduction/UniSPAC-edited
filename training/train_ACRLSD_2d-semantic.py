import gc
import itertools
import logging
import os
import random

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
from training.models.ACRLSD import ACRLSD_2D, remove_module
from training.models.segEM import segEM2d
from training.utils.aftercare import normalize_affinity, visualize_and_save_affinity, aftercare
from training.utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train, load_train_dataset, collate_fn_2D_fib25_Train, \
    get_acc_prec_recall_f1, load_semantic_dataset
from utils.dataloader_ninanjie import Dataset_2D_ninanjie_Train
from utils.dataloader_hemi_better import Dataset_2D_hemi_Train, collate_fn_2D_hemi_Train


# from utils.dataloader_cremi import Dataset_2D_cremi_Train,collate_fn_2D_cremi_Train

## CUDA_VISIBLE_DEVICES=1 python train_ACRLSD_2d.py &

def set_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministics = True


class segEM2d_cell_seg(torch.nn.Module):
    def __init__(self):
        super(segEM2d_cell_seg, self).__init__()
        self.acrlsd = ACRLSD_2D(num_fmaps=32, fmap_inc_factors=5,
                                downsample_times=3)
        self.acrlsd.load_state_dict(
            remove_module(torch.load('./output/log/ACRLSD_2D(ninanjie)_all/256_32_5_3/auto_unweighted/Best_in_val.model')))
        for param in self.acrlsd.parameters():
            param.requires_grad = False
        self.cell_seg = segEM2d(num_fmaps=24, fmap_inc_factors=4,
                                downsample_times=4)
        params = remove_module(torch.load('./output/log/segEM2d(ninanjie)-prompt-3rd-1/weighted-w2-5-w3-1/24_3_4/Best_in_val.model'))
        params = {k: v for k, v in params.items() if 'affinity' not in k}
        self.cell_seg.load_state_dict(params)
        for param in self.cell_seg.parameters():
            param.requires_grad = False

    def forward(self, x, x_prompt):
        with torch.no_grad():
            _, y_affinity = self.acrlsd(x)
            y_mask = self.cell_seg(x_prompt, y_affinity)
        return y_mask


def model_step(model, loss_fn, optimizer, raw, labels, gt_lsds, gt_affinity, activation,
               class_idx=-1, scalar=None, train_step=True, auto_loss=False):
    if train_step:
        optimizer.zero_grad()

    # forward
    with torch.cuda.amp.autocast(enabled=scalar is not None):
        lsd_logits, affinity_logits = model(raw)
        if class_idx != -1:
            labels = torch.as_tensor(labels, device=device)
            class_mask = (labels != class_idx).float().unsqueeze(1)
            affinity_logits = affinity_logits * (1 + class_mask * 5.0)
            lsd_logits = lsd_logits * (1 + class_mask * 5.0)
        loss_1, loss_2 = loss_fn(lsd_logits, gt_lsds), loss_fn(affinity_logits, gt_affinity)
        if auto_loss:
            loss_value = loss_1 / loss_1.detach() + loss_2 / loss_2.detach()
        else:
            loss_value = loss_1 + loss_2

    if train_step:
        if scalar is not None:
            scalar.scale(loss_value).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            loss_value.backward()
            optimizer.step()

    affinity_output = activation(affinity_logits)

    outputs = {
        'pred_affinity': affinity_output,
        'affinity_logits': affinity_logits,
    }

    return loss_1.detach().item() + loss_2.detach().item(), outputs


def weighted_model_step(model, loss_fn, optimizer, raw, labels, gt_lsds, gt_affinity, activation,
                        class_idx=-1, scalar=None, train_step=True, auto_loss=False):
    if train_step:
        optimizer.zero_grad()

    with torch.cuda.amp.autocast(enabled=scalar is not None):
        lsd_logits, affinity_logits = model(raw)

        fg_mask_affinity = (gt_affinity > 0).float()  # 亲和度前景掩码
        fg_mask_lsds = (gt_lsds > 0).float()  # LSDS前景掩码（根据实际标签调整）

        fg_pixels_affinity = fg_mask_affinity.sum()
        bg_pixels_affinity = fg_mask_affinity.numel() - fg_pixels_affinity
        fg_weight_affinity = bg_pixels_affinity / (fg_pixels_affinity + 1e-5) if fg_pixels_affinity > 0 else 2.0

        fg_pixels_lsds = fg_mask_lsds.sum()
        bg_pixels_lsds = fg_mask_lsds.numel() - fg_pixels_lsds
        fg_weight_lsds = bg_pixels_lsds / (fg_pixels_lsds + 1e-5) if fg_pixels_lsds > 0 else 2.0

        weights_affinity = torch.where(torch.as_tensor(fg_mask_affinity == 1.0), fg_weight_affinity, torch.tensor(1.0, device=device)).to(device)
        weights_lsds = torch.where(torch.as_tensor(fg_mask_lsds == 1.0), fg_weight_lsds, torch.tensor(1.0, device=device)).to(device)

        if class_idx != -1:
            labels = torch.as_tensor(labels, device=device)
            class_mask = (labels != class_idx).float().unsqueeze(1)
            weights_affinity = weights_affinity * (1 + class_mask * 5.0)
            weights_lsds = weights_lsds * (1 + class_mask * 5.0)

        def weighted_loss(pred, target, weights):
            base_loss = loss_fn(pred, target)
            weighted = base_loss * weights
            return weighted.mean()

        loss_lsd = weighted_loss(lsd_logits, gt_lsds, weights_lsds)
        loss_affinity = weighted_loss(affinity_logits, gt_affinity, weights_affinity)

        if auto_loss:
            loss_value = loss_lsd / loss_lsd.detach() + loss_affinity / loss_affinity.detach()
        else:
            loss_value = loss_lsd + loss_affinity

    # 反向传播与参数更新
    if train_step:
        if scalar is not None:
            scalar.scale(loss_value).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
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
    return loss_lsd.detach().item() + loss_affinity.detach().item(), outputs


def trainer(num_fmaps, fmap_inc_factors, class_idx=1, weighted=False, auto_loss=True, other_class_punish=True):
    Save_Name = (f'ACRLSD_2D(ninanjie)_semantic/{class_idx}/{crop_size}_{num_fmaps}_{fmap_inc_factors}/'
                 f'{"weighted" if weighted else "weightless"}_{"auto" if auto_loss else "normal"}{"_punished" if other_class_punish else ""}')
    os.makedirs('./output/log/' + Save_Name, exist_ok=True)
    for file in os.listdir('./output/log/' + Save_Name):
        os.remove(os.path.join('./output/log/' + Save_Name, file))

    model = ACRLSD_2D(num_fmaps=num_fmaps, fmap_inc_factors=fmap_inc_factors, downsample_times=3)
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    # cell_seg.load_state_dict(
    #     remove_module(torch.load('./output/log/segEM2d(ninanjie)-prompt-3rd/unweighted-auto-256_32_5_3/Best_in_val.model')))
    # for param in cell_seg.parameters():
    #     param.requires_grad = False

    scaler = torch.cuda.amp.GradScaler()

    ##创建log日志
    writer = SummaryWriter(log_dir='./output/log/' + Save_Name)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logfile = './output/log/{}/log.txt'.format(Save_Name)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.MSELoss().to(device)
    model_step_fn = weighted_model_step if weighted else model_step

    # training loop
    model.train()
    loss_fn.train()
    epoch = 0
    Best_loss = 1e5
    Best_epoch = 15
    early_stop_count = 35
    no_improve_count = 0
    with tqdm(total=training_epochs) as pbar:
        progress_desc, train_desc, val_desc = '', '', ''
        while epoch < training_epochs:
            ###################Train###################
            model.train()
            # reset data loader to get random augmentations
            np.random.seed()
            random.seed()
            count, total = 0, len(train_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in train_loader:
                ##Get Tensor
                # raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                # gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                # gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                #                               device=device)  # (batch, 2, height, width)
                count += 1
                progress_desc = f"Train {count}/{total}, "
                model_step_fn(model, loss_fn, optimizer, raw, labels, gt_lsds, gt_affinity, activation,
                              class_idx=class_idx if other_class_punish else -1, scalar=scaler, train_step=True, auto_loss=auto_loss)
                pbar.set_description(progress_desc + train_desc + val_desc)

            ###################Validate###################
            model.eval()
            ##Fix validation set
            seed = 98
            np.random.seed(seed)
            random.seed(seed)
            voi_val, val_loss = 0, 0
            metrics = np.array([0, 0, 0, 0], dtype=np.float32)
            count, total = 0, len(val_loader)
            # for raw, labels, Points_pos,Points_lab,Boxes,point_map,mask,gt_affinity,gt_lsds in tmp_val_loader:
            for raw, labels, point_map, mask, gt_affinity, gt_lsds in val_loader:
                # raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
                # gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
                # gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                #                               device=device)  # (batch, 2, height, width)
                with torch.no_grad():
                    loss_value, output = model_step_fn(model, loss_fn, optimizer, raw, labels, gt_lsds, gt_affinity,
                                                       activation, scalar=scaler, train_step=False, auto_loss=auto_loss,
                                                       class_idx=class_idx if other_class_punish else -1)
                    raw_mask = raw
                    y_affinity = np.asarray(output['pred_affinity'].detach().cpu())
                    binary_y_pred = np.asarray(y_affinity * 255.0, dtype=np.uint8)
                    binary_gt_affinity = np.asarray(gt_affinity.detach().cpu() * 255.0, dtype=np.uint8)

                count += 1
                # metrics += np.asarray(get_acc_prec_recall_f1(binary_y_pred.flatten(), binary_gt_affinity.flatten()))
                voi_val += np.sum(variation_of_information(binary_gt_affinity.flatten(), binary_y_pred.flatten()))
                # for class_id in range(1):
                #     voi_class[class_id] += np.sum(variation_of_information(
                #         binary_gt_affinity[:, [class_id, class_id + 1], ...].flatten(),
                #         binary_y_pred[:, [class_id, class_id + 1], ...].flatten()
                #     ))
                #     metrics_class[class_id] += np.asarray(
                #         get_acc_prec_recall_f1(
                #             binary_y_pred[:, [class_id, class_id + 1], ...].flatten(),
                #             binary_gt_affinity[:, [class_id, class_id + 1], ...].flatten()
                #         )
                #     )
                val_loss += loss_value
                progress_desc = f"Val {count}/{total}, "
                pbar.set_description(progress_desc + train_desc + val_desc)

            val_loss /= (total + 0.)
            voi_val /= (total + 0.)
            metrics /= (total + 0.)
            ###################Compare###################
            if Best_loss > val_loss:
                Best_loss = val_loss
                Best_epoch = epoch
                torch.save(model.state_dict(), './output/log/{}/Best_in_val.model'.format(Save_Name))
                no_improve_count = 0
            else:
                no_improve_count += 1

            val_desc = f'best {Best_epoch}, VOI: {voi_val:.5f}'

            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/voi', voi_val, epoch)
            # for class_id in range(1):
            #     writer.add_scalar(f'voi/class_{class_id}', voi_class[class_id], epoch)
            #     writer.add_scalar(f'metrics/acc_class_{class_id}', metrics_class[class_id][0], epoch)
            #     writer.add_scalar(f'metrics/prec_class_{class_id}', metrics_class[class_id][1], epoch)
            #     writer.add_scalar(f'metrics/recall_class_{class_id}', metrics_class[class_id][2], epoch)
            #     writer.add_scalar(f'metrics/f1_class_{class_id}', metrics_class[class_id][3], epoch)

            random.seed(epoch)
            sample_idx = random.randint(0, raw.shape[0] - 1)
            raw_np = raw_mask[sample_idx, ...].detach().cpu().numpy()
            y_affinity_np = y_affinity[sample_idx, ...] * 255.  # (8, height, width)
            gt_affinity_np = gt_affinity[sample_idx, ...].detach().cpu().numpy() * 255.
            visual_y_affinities = [normalize_affinity(np.sum(y_affinity_np[[idx, idx + 1], ...], axis=0))
                                   for idx in range(0, y_affinity_np.shape[0]-1, 2)]
            visual_gt_affinities = [normalize_affinity(np.sum(gt_affinity_np[[idx, idx + 1], ...], axis=0))
                                    for idx in range(0, y_affinity_np.shape[0]-1, 2)]
            visualize_and_save_affinity(raw_np, epoch, 'visual/raw', writer)
            if class_idx == -1:
                for idx in range(3):
                    visualize_and_save_affinity(visual_y_affinities[idx], epoch, f'pred/class_{idx}', writer)
                    visualize_and_save_affinity(visual_gt_affinities[idx], epoch, f'gt/class_{idx}', writer)
            else:
                visualize_and_save_affinity(visual_y_affinities[0], epoch, f'visual/pred', writer)
            writer.flush()

            pbar.update(1)
            epoch += 1

            ##Early stop
            if no_improve_count > early_stop_count and epoch > 100:
                logging.info("Early stop!")
                break
    fh.flush()
    torch.save(model.state_dict(), './output/log/{}/final.model'.format(Save_Name))


if __name__ == '__main__':
    ##设置超参数
    training_epochs = 10000
    learning_rate = 1e-4
    batch_size = 30

    set_seed()

    ###创建模型
    # set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,1,5,0,3'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpus = [i for i in range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))]

    cell_seg = segEM2d_cell_seg()
    cell_seg = torch.nn.DataParallel(cell_seg.cuda(), device_ids=gpus, output_device=gpus[0])

    dataset_names = ['second_6', 'fourth_1']
    class_idx = 1
    crop_size = 512
    crop_xyz = [5, 4, 1]
    train_dataset, val_dataset = [], []
    for dataset_name, i, j, k in itertools.product(
            dataset_names, range(crop_xyz[0]), range(crop_xyz[1]), range(crop_xyz[2])
    ):
        train_tmp, val_tmp = load_semantic_dataset(dataset_name, raw_dir='raw', label_dir='export',
                                                   require_xz_yz=False, from_temp=False, crop_size=crop_size,
                                                   crop_xyz=crop_xyz, chunk_position=[i, j, k], class_idx=class_idx)
        train_dataset.append(train_tmp)
        val_dataset.append(val_tmp)

    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=28,
                              pin_memory=True, drop_last=True, collate_fn=collate_fn_2D_fib25_Train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=28,
                            pin_memory=True, collate_fn=collate_fn_2D_fib25_Train)

    def load_data_to_device(loader, cell_seg):
        res = []
        for raw, labels, point_map, mask, gt_affinity, gt_lsds in loader:
            raw = torch.as_tensor(raw, dtype=torch.float, device=device)  # (batch, 1, height, width)
            with torch.no_grad():
                cell_seg.eval()
                y_mask = (cell_seg(raw, torch.ones_like(raw).squeeze()) > 0.5) + 0.  # (batch, 1, height, width)
                y_mask = y_mask.permute(1, 0, 2, 3).unsqueeze(4)  # (batch, height, width)
                y_mask = torch.as_tensor(aftercare(y_mask)).squeeze().unsqueeze(1).to(device)  # (batch, 1, height, width)
                raw = raw * y_mask  # (batch, 1, height, width)
            gt_lsds = torch.as_tensor(gt_lsds, dtype=torch.float, device=device)  # (batch, 6, height, width)
            gt_affinity = torch.as_tensor(gt_affinity, dtype=torch.float,
                                          device=device)  # (batch, 2, height, width)
            res.append((raw, labels, point_map, mask, gt_affinity, gt_lsds))
        return res

    train_loader, val_loader = load_data_to_device(train_loader, cell_seg), load_data_to_device(val_loader, cell_seg)

    del cell_seg
    torch.cuda.empty_cache()
    gc.collect()

    # trainer(30, 5, class_idx=class_idx, weighted=True, auto_loss=True, other_class_punish=True)
    trainer(32, 5, class_idx=class_idx, weighted=False, auto_loss=True, other_class_punish=True)
    # trainer(30, 5, class_idx=class_idx, weighted=True, auto_loss=True, other_class_punish=False)
    # trainer(30, 5, class_idx=class_idx, weighted=False, auto_loss=True, other_class_punish=False)
    # trainer(30, 5, class_idx=class_idx, weighted=False, auto_loss=False, other_class_punish=True)
