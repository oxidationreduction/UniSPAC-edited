import multiprocessing
import os

import imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from training.utils.dataloader_ninanjie import load_test_dataset, Dataset_2D_ninanjie_Origin, \
    collate_fn_2D_ninanjie_Origin, ninanjie_data
from utils.dataloader_ninanjie import collate_fn_2D_fib25_Test

## CUDA_VISIBLE_DEVICES=0 python main_segEM_2d_train_zebrafinch.py &

HOME_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def visualize_and_save_mask(raw, mask, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98):
    from skimage.measure import label
    output_path = os.path.join(output_dir, f"{mode}_{str(idx).zfill(5)}.tif")

    # mask = mask > 127
    # segmentation = label(np.asarray(0 + mask)[:, :])

    segmentation = mask.copy()

    raw[mask == 0] = 1

    # 获取所有实例标签（排除背景0）
    instances = np.unique(segmentation)
    instances = instances[instances != 0]
    num_instances = len(instances)

    # 设置随机种子，确保颜色分配一致
    np.random.seed(seed)

    # 为每个实例生成随机颜色（RGB格式，值在0-1之间）
    # 避免过暗颜色，提高可视性
    colors = np.random.rand(num_instances, 3)
    colors = np.clip(colors, 0.4, 0.9)  # 限制颜色亮度范围

    # 创建颜色映射：索引0对应背景，其余对应各个实例
    # 颜色映射的索引需要与segmentation中的标签值对应
    max_label = int(np.max(segmentation)) if num_instances > 0 else 0
    cmap_colors = [background_color] * (max_label + 1)  # 初始化所有标签颜色

    # 为每个实例标签分配颜色
    for i, label in enumerate(instances):
        cmap_colors[label] = tuple(colors[i])

    # 创建自定义颜色映射
    cmap = ListedColormap(cmap_colors)

    # 绘制并保存图像
    plt.figure(figsize=(10, 10))
    # 显示原始图像
    plt.imshow(raw, cmap='gray' if raw.ndim == 2 else None)
    # 显示分割结果（仅显示非背景区域）
    masked_segmentation = np.ma.masked_where(segmentation == 0, segmentation)
    plt.imshow(masked_segmentation, cmap=cmap, alpha=0.5)  # alpha控制透明度
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)  # 去除边距

    # 保存图像，不包含白边
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    return 1


def save_mask(mask, idx=0):
    from skimage.measure import label
    output_path = os.path.join(output_dir, f"{str(idx).zfill(4)}.tif")

    mask = mask > 127
    segmentation = np.asarray(label(np.asarray(0 + mask)[:, :])).astype(np.uint16)

    imageio.imwrite(output_path, segmentation)
    return 1


def save_dataset(raw, labels):
    pool = multiprocessing.Pool(os.cpu_count() >> 2)
    processes = []
    for idx, (_, single_mask) in enumerate(zip(raw, labels)):
        single_mask = single_mask.squeeze().astype(np.uint16)
        processes.append(pool.apply_async(save_mask,
                                          args=(single_mask, idx + count * batch_size)))
    with tqdm(total=len(processes), leave=False, desc="saving") as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()


def visualize_dataset(raw, labels):
    pool = multiprocessing.Pool(os.cpu_count() >> 2)
    processes = []
    for idx, (single_raw, single_mask) in enumerate(zip(raw, labels)):
        single_raw = single_raw.squeeze()
        single_mask = single_mask.squeeze().astype(np.uint16)
        processes.append(pool.apply_async(visualize_and_save_mask,
                                              args=(single_raw, single_mask, idx + count * batch_size)))

    with tqdm(total=len(processes), leave=False, desc="saving") as pbar:
        for process in processes:
            res = process.get()
            pbar.update(res)
    pool.close()
    pool.join()


if __name__ == '__main__':
    batch_size = 256
    dataset_name = 'best_val_3'
    raw_dirs = ['raw_2']
    # label_dirs = ['train_probability_tz']
    label_dirs = ['truth_label_2_seg_1']
    for raw_dir in raw_dirs:
        for label_dir in label_dirs:
            test_dataset = Dataset_2D_ninanjie_Origin(data_dir=os.path.join(ninanjie_data, 'train'),
                                                      raw_dir=raw_dir, label_dir=label_dir,
                                                      batch_num=dataset_name,
                                                      crop_xyz=[1,1,1], chunk_position=[0,0,0])
            origin_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=48,
                                       pin_memory=True, drop_last=False, collate_fn=collate_fn_2D_fib25_Test)

            # output_dir = os.path.join(HOME_PATH, f'data/ninanjie/train/{dataset_name}/label_2')
            output_dir = os.path.join(HOME_PATH, f'data/ninanjie/train/{dataset_name}/visual/{label_dir}')
            os.makedirs(output_dir, exist_ok=True)

            count = 0
            for raw, _, labels in origin_loader:
                # save_dataset(raw, labels)
                visualize_dataset(raw, labels)
                count += 1
