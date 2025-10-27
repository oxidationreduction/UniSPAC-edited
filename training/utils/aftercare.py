import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage import measure
from torch.nn import functional as F


def aftercare(y_mask: torch.Tensor):
    """
    对生物组织图像分割结果进行后处理

    参数:
        y_mask: 输入的分割结果张量，只含0和1
                可以是(B, H, W)的单层分割结果
                也可以是(B, N, H, W)的多层连续分割结果

    返回:
        处理后的分割结果张量
    """
    # 确保输入是浮点型张量
    y_mask = y_mask + 0. # (1, B, H, W, N)
    if y_mask.ndim == 5:
        y_mask = y_mask.squeeze(0)  # (B, H, W, N)
    device = torch.device('cpu')
    batch_size = y_mask.size(0)

    # 根据输入形状确定是否为多层结构
    # is_multi_layer = len(y_mask.shape) >= 4  # (B, N, H, W)

    # 定义结构元素（3x3的十字形结构，适合生物组织的连接性）
    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], device=device, dtype=torch.float32)
    kernel = kernel.view(1, 1, 3, 3)  # 适配卷积操作的形状 [out_channels, in_channels, kH, kW]

    # 存储处理结果
    result = []

    # 对每个样本进行处理
    for b in range(batch_size):
        # 多层结构: (B, N, H, W)
        y_mask_ = y_mask[b].permute(2, 0, 1)  # (N, H, W)
        layers = []
        for n in range(y_mask_.size(0)):
            layer = y_mask_[n]   # [H, W]
            processed_layer = process_single_layer(layer, device=device, kernel=kernel, erosion=True) # (1, H, W)
            layers.append(processed_layer)
        # 堆叠处理后的多层
        chunk = torch.cat(layers, dim=0).to(device) # (N, H, W)
        chunk = torch.as_tensor(measure.label(np.asarray(chunk))) # relabel
        layers = []
        for n in range(chunk.size(0)):
            layer = y_mask_[n]  # [H, W]
            processed_layer = process_single_layer(layer, device=device, kernel=kernel, erosion=False)  # (1, H, W)
            layers.append(processed_layer)
        chunk = torch.cat(layers, dim=0).to(device).permute(1, 2, 0).unsqueeze(0) # (1, H, W, N)
        result.append(chunk) # (1, H, W, N) each

    # 堆叠所有样本
    result = torch.cat(result, dim=0).unsqueeze(0) # (1, B, H, W, N)

    # 确保输出仍是0和1的二进制张量
    return np.asarray(result.cpu())


def process_single_layer(layer: torch.Tensor, device, kernel, erosion=False):
    """处理单个图层，执行填充空心和圆化操作"""
    layer = np.asarray(layer.cpu()) # (H, W)

    min_size = 1000
    labels, counts = np.unique(layer, return_counts=True)
    label_counts = dict(zip(labels, counts))
    layer_new = torch.zeros_like(torch.as_tensor(layer), dtype=torch.int32, device=device) # (H, W)

    morphological_op = morphological_erosion if erosion else morphological_dilation

    if 0.0 in label_counts.keys():
        label_counts.pop(0.0)
    for label, count in label_counts.items():  # 跳过背景0
        if count > min_size:
            mask = torch.as_tensor(binary_fill_holes(layer == label) + 0., dtype=torch.float32, device=device).unsqueeze(0)
            # mask = morphological_dilation(mask, kernel) # (1, H, W)
            mask = morphological_op(mask, kernel)
            mask = (mask == 1.).squeeze(0) # (H, W)
            layer_new[mask] = 1

    return layer_new.unsqueeze(0)


def morphological_dilation(x: torch.Tensor, kernel: torch.Tensor):
    """形态学膨胀操作"""
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)  # SAME padding
    x_dilated = F.conv2d(x, kernel, padding=padding)
    return (x_dilated > 0.5).float()  # 二值化


def morphological_erosion(x: torch.Tensor, kernel: torch.Tensor):
    """形态学腐蚀操作"""
    padding = (kernel.size(2) // 2, kernel.size(3) // 2)  # SAME padding
    # 腐蚀是对反相图像的膨胀
    x_inverted = 1 - x
    x_eroded_inverted = F.conv2d(x_inverted, kernel, padding=padding)
    x_eroded = 1 - (x_eroded_inverted > 0.5).float()  # 反相回来并二值化
    return x_eroded


def morphological_closing(x: torch.Tensor, kernel: torch.Tensor, iterations: int = 1):
    """形态学闭运算（先膨胀后腐蚀）"""
    result = x
    for _ in range(iterations):
        result = morphological_dilation(result, kernel)
        result = morphological_erosion(result, kernel)
    return result


def visualize_and_save_mask(raw: torch.Tensor, segmentation: torch.Tensor, idx=0, mode='origin',
                            background_color=(0, 0, 0), seed=98, writer=None):
    """
    可视化原始图像和分割掩码，并可选择保存为图片或写入TensorBoard

    参数:
        raw: 原始图像
        segmentation: 分割结果
        idx: 图像索引，用于命名
        mode: 可视化模式
        background_color: 背景颜色
        seed: 随机种子，确保颜色一致性
        writer: TensorBoard的SummaryWriter实例，如果提供则写入TensorBoard
        tag: TensorBoard中的标签
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    # After this if-else block, raw is (H, W), segmentation is (H, W) with labeled instances
    raw = np.asarray(raw).squeeze()
    if raw.ndim == 4:
        raw = raw.transpose(0, 3, 1, 2)[1, 0, ...].squeeze()
        segmentation = np.asarray(segmentation, dtype=np.int32).squeeze().transpose(0, 3, 1, 2)[1, 0, ...].squeeze()
        segmentation = measure.label(segmentation)
    elif raw.ndim == 3:
        raw = raw[1, ...].squeeze()
        segmentation = np.asarray(segmentation, dtype=np.int32).squeeze()[1, ...].squeeze()
        segmentation = measure.label(segmentation)
    elif raw.ndim == 2:
        raw = raw.squeeze()
        segmentation = np.asarray(segmentation, dtype=np.int32).squeeze()
        segmentation = measure.label(segmentation)
    else:
        raise AssertionError('unexpected raw shape')

    # 处理模式为'seg_only'的情况
    if 'seg_only' in mode:
        mask = segmentation > 0
        raw = raw * mask

    # 获取所有实例标签（排除背景0）
    instances = np.unique(segmentation)
    instances = instances[instances != 0]
    num_instances = len(instances)

    # 设置随机种子，确保颜色分配一致
    np.random.seed(seed)

    # 为每个实例生成随机颜色（RGB格式，值在0-1之间）
    # 避免过暗颜色，提高可视性
    colors = np.random.rand(num_instances, 3)
    colors = np.clip(colors, 0.5, 0.9)  # 限制颜色亮度范围

    # 创建颜色映射：索引0对应背景，其余对应各个实例
    max_label = int(np.max(segmentation)) if num_instances > 0 else 0
    cmap_colors = [background_color] * (max_label + 1)  # 初始化所有标签颜色

    # 为每个实例标签分配颜色
    for i, label in enumerate(instances):
        cmap_colors[label] = tuple(colors[i])

    # 创建自定义颜色映射
    cmap = ListedColormap(cmap_colors)

    # 绘制图像
    plt.figure(figsize=(10, 10))
    # 显示原始图像
    plt.imshow(raw, cmap='gray' if raw.ndim == 2 else None)
    # 显示分割结果（仅显示非背景区域）
    masked_segmentation = np.ma.masked_where(segmentation == 0, segmentation)
    plt.imshow(masked_segmentation, cmap=cmap, alpha=0.5)  # alpha控制透明度
    # 获取当前坐标轴，用于添加圆形和文本
    ax = plt.gca()

    # 为每个实例添加中心红色圆形和编号
    for label in instances:
        # 找到该实例的所有像素坐标
        coords = np.where(segmentation == label)

        # 计算中心坐标（平均位置）
        if len(coords[0]) > 0:  # 确保实例有像素
            center_y = int(np.mean(coords[0]))
            center_x = int(np.mean(coords[1]))

            # 添加红色圆形
            circle = plt.Circle((center_x, center_y), radius=20, color='red', fill=True)
            ax.add_patch(circle)

            # 添加数字编号（白色文本，居中显示）
            plt.text(center_x, center_y, str(int(label)),
                     color='white', fontsize=35,
                     ha='center', va='center',
                     fontweight='bold')

    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)  # 去除边距

    # 如果提供了writer，则将图像写入TensorBoard
    if writer is not None:
        # 将matplotlib图像转换为numpy数组以便TensorBoard显示
        import io
        from PIL import Image

        # 将图像保存到内存缓冲区
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        buf.seek(0)

        # 转换为PIL图像并再转换为numpy数组
        img = Image.open(buf)
        img_array = np.array(img)

        # 写入TensorBoard，注意需要调整通道顺序为[C, H, W]
        if img_array.ndim == 3:
            img_array = img_array.transpose(2, 0, 1)

        # 添加到TensorBoard，使用idx作为全局步长
        writer.add_image(mode, img_array, global_step=idx, dataformats='CHW')

    plt.close()

def visualize_and_save_affinity(affinity: np.ndarray, idx=0, mode='origin', writer=None):
    """
    可视化亲和力图，并可选择保存为图片或写入TensorBoard
    参数:
        affinity: 亲和力图张量
        idx: 图像索引，用于命名
        mode: 可视化模式
        writer: TensorBoard的SummaryWriter实例，如果提供则写入TensorBoard
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    import io
    from PIL import Image

    # 将图像保存到内存缓冲区
    plt.imshow(affinity.squeeze(), cmap='gray')
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)  # 去除边距

    buf = io.BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)

    # 转换为PIL图像并再转换为numpy数组
    img = Image.open(buf)
    img_array = np.array(img)

    # 写入TensorBoard，注意需要调整通道顺序为[C, H, W]
    if img_array.ndim == 3:
        img_array = img_array.transpose(2, 0, 1)

    # 添加到TensorBoard，使用idx作为全局步长
    writer.add_image(mode, img_array, global_step=idx, dataformats='CHW')
    plt.close()


def normalize_affinity(affinity: np.ndarray) -> np.ndarray:
    """Normalize affinity values to the range [0, 1]."""
    min_val = np.min(affinity)
    max_val = np.max(affinity)
    if min_val == max_val:
        return np.zeros_like(affinity)
    normalized = (affinity - min_val) / (max_val - min_val)
    return normalized


def calculate_batch_voi(preds, gts):
    """
    批量计算VOI.
    preds 可能是 4 维或以上的数据，需要转换为 3 维度才能使用 variation of information 计算。无论维度多少，都要使用循环，最内层循环计算 3 维 VOI。
    参数:
        preds: 预测标签张量
        gts: 真实标签张量
    返回:
        mean_voi: 平均VOI值
    """
    preds = torch.as_tensor(preds, dtype=torch.long)
    gts = torch.as_tensor(gts, dtype=torch.long)
    assert preds.shape[0] == gts.shape[0], "批量大小不匹配"
    batch_size = preds.shape[0]
    voi_list = []

    # 遍历批量中的每个样本
    for i in range(batch_size):
        pred = preds[i]
        gt = gts[i]

        # 如果有多余维度，压缩到3维 (H, W, N)
        if pred.ndim > 3:
            pred = pred.permute(1, 2, 0).contiguous()  # (H, W, N)
        if gt.ndim > 3:
            gt = gt.permute(1, 2, 0).contiguous()  # (H, W, N)

        # 对每个通道计算VOI并取平均
        channel_voi = []
        for n in range(pred.shape[-1]):
            voi = calculate_multi_class_voi(pred[..., n], gt[..., n])
            channel_voi.append(voi)
        mean_channel_voi = np.mean(channel_voi) if channel_voi else 0.0
        voi_list.append(mean_channel_voi)

    mean_voi = np.mean(voi_list) if voi_list else 0.0
    return voi_list, mean_voi


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
