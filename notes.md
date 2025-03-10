## UniSPAC-2D

### 模型结构

亲和度预测：ACRLSD（预训练，UniSPAC-2D 训练时锁定）
掩码预测：2D-UNet 和 Conv2d

### 数据流

0. 输入：原数据 `x_raw`（1 通道灰度图） 和 prompt 数据 `x_prompt`（1 通道）
1. `ACRLSD(x_raw)` -> `y_lsds`（6 通道）, `y_affinity`（2 通道），分别是 local shape descriptors (LSDs，作为中间结果用来提升亲和度预测质量) 和 affinity
2. `2D-UNet*Conv2d([x_prompt, y_affinity])` -> `y_mask`（2 通道），神经元掩码预测

### ACRLSD

用于输出 LSDs 和亲和度的模型，输入原始数据，输出 LSDs 和亲和度。

LSDs 预测：2D-UNet，1 通道输入，6 通道输出。
亲和度预测：2D-UNet，7 通道输入（原始数据 x 和 LSDs 进行拼接），2 通道输出。

0. 输入：原数据 `x`（1 通道灰度图）
1. `lsd_predict(x)` -> `y_lsds`（6 通道）
2. `affinity_predict([x, y_lsds])` -> `y_affinity`（2 通道）

### 训练数据

#### Prompt 生成（get_prompt）

一半数据有 prompt，一半没有，随机确定是否生成。

如果生成 prompt，随机决定对多少个神经元进行 prompt，factor 从 0.1 到 0.9 随机确定。

**生成正样本点**：对于每个包含的 label，生成该类别的 mask_label，与全局 mask 做或运算。在此 label 区域内随机选择一个点作为正样本点，添加到 Points_pos 中，label=1 添加到 Points_lab 中。最后计算此 label 的边界 Box，添加到 Boxes 中。

**生成负样本点**：对于每个排除的 label，，在此 label 区域内随机选择一个点作为负样本点，添加到 Points_pos 中。label=0 添加到 Points_lab 中。

#### Prompt Map（generate_gaussial_matrix）

将 prompt 点转化为 map，利用高斯分布确定像素点的 prompt 值。

1. 无 prompt：分割所有神经元
2. 存在正样本点：只分割正样本点所在的神经元
3. 只存在负样本点：分割所有神经元，但是不分割负样本点所在的神经元

### 三种训练模式

1. 基本款：包含 binary_cross_entropy 和 DiceLoss
2. 增强款：用于跨数据种类场景，在基本款基础上增加 `loss3 = torch.sum(y_mask * gt_affinity) / torch.sum(gt_affinity)`，引导模型在真实亲和力区域内产生更高的预测值，能显著提高神经元边界的预测质量
3. 持续学习款：ACRLSD 不从预训练数据加载，也参与训练。在增强款基础上增加 `loss_affinity`，使用 MSEloss 计算 LSDs 和亲和度预测各自的损失。使用 avalanche 持续学习库，依次学习各部分数据。
