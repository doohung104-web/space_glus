# GLUS 轨迹分支技术设计文档（贴合当前仓库实现）

仓库：`GLUS-video/GLUS`  
面向版本：当前公开仓库主线结构  
设计目标：为 GLUS 增加一个**可开关的 trajectory 分支**，在不破坏原始训练 / 推理主流程的前提下，为 `[SEG]` 对应的分割提示引入显式运动先验。

---

# 1. 设计原则

## 1.1 总目标

在 GLUS 现有结构上加入一个**trajectory-aware 辅助分支**，使模型在处理视频 referring segmentation 时，除了依赖：

- context frames 的全局语义
- 当前 question frame 的局部视觉特征
- memory bank 的时序传播

还能够显式利用：

- 目标历史 mask 所推导出的**低维运动状态序列**

这些运动状态不替代原始 GLUS 主路径，而是作为**附加条件**影响 `[SEG]` 的表示，最终改善 mask 解码。

---

## 1.2 必须满足的约束

该分支必须满足以下工程约束：

1. **默认关闭**
   - 所有新逻辑受 `--use_trajectory` 控制
   - 当 `False` 时，应尽量保持当前训练 / 推理行为与原仓库一致

2. **不删除、不重写原始损失**
   - 保留原始：
     - `ce_loss`
     - `mask_bce_loss`
     - `mask_dice_loss`
     - `contrastive_loss`（若启用）
   - trajectory 分支仅在启用时额外叠加辅助项

3. **不破坏原始 `[SEG] -> SAM2` 主链**
   - 仍然以 `[SEG]` hidden state 投影为 segmentation embedding
   - trajectory 只作为对 `[SEG]` embedding 的附加调制，不替换原逻辑

4. **尽量避免大规模修改 tokenizer / prompt 机制**
   - GLUS 当前 prompt、`seg_token_mask`、`offset`、多 conversation 对齐方式都较手工
   - 第一版不建议直接引入复杂 token hook 注入机制

---

## 1.3 本设计的核心取舍

相较于“新增 `[TRAJ_1]...[TRAJ_4]` 特殊 token 并在 embedding 层注入”的方案，本设计优先采用：

> **先做“非侵入式 trajectory 条件调制”**：  
> 在 `[SEG]` embedding 被抽取之后、送入 SAM2 prompt encoder 之前，用 trajectory encoder 输出的向量对其进行 residual modulation。

原因：

- 更贴合当前 `model/GLUS.py` 的实现方式
- 不破坏 `seg_token_mask` 与 conversation token 对齐逻辑
- 不需要处理 `generate()` / `forward()` 下 token hook 的兼容性
- 更适合做第一阶段可验证 MVP

---

# 2. 当前 GLUS 代码结构中的接入点

---

## 2.1 训练主路径

当前训练路径：

```text
train_ds.py
  -> HybridDataset / ReferVideoSegDataset
  -> collate_fn
  -> model(**input_dict)
  -> GLUSForCausalLM.generate_masks()
  -> 提取 [SEG] hidden states
  -> SAM2 prompt encoder / mask decoder
  -> loss
```

trajectory 分支最适合插入的位置：

### 数据侧
- `dataset/refer_video_seg_dataset.py`
- `dataset/dataset.py`

### 模型侧
- `model/trajectory_module.py`（新增）
- `model/GLUS.py`

### 参数入口
- `train_ds.py`
- `inference_iter.py`

---

## 2.2 推理主路径

当前推理路径：

```text
inference_iter.py
  -> inference_video()
  -> inference_frames()
  -> model.evaluate()
  -> generate()
  -> 提取 [SEG] embedding
  -> SAM2 decode
  -> 保存 mask
```

trajectory 分支推理接入点：

- `inference_iter.py` 中维护历史预测 mask 滑窗
- 从预测 mask 提取状态
- 将轨迹状态传给 `model.evaluate()`
- 在 `evaluate()` 中调制 `[SEG]` embedding

---

## 2.3 为什么不优先改 prompt / 新 token

当前 GLUS 的关键脆弱点有：

- `utils/utils.py` 中 `CONTEXT_FRAME_NUM`, `QUESTION_FRAME_NUM` 常量和 prompt 耦合
- `[SEG]` 的定位通过手工构造 `seg_token_mask`
- 一个 batch 内有多个 expressions / conversations，依赖 `offset` 做对齐
- `generate_masks()` 和 `evaluate()` 的 token 路径不同

因此第一版 trajectory 设计不建议：

- 增加 `[TRAJ_1]...[TRAJ_4]`
- 在文本 prompt 中插大块结构化 token
- 在 `embed_tokens` 上挂 hook

这些都适合后续增强版，而不是第一版。

---

# 3. 方案概述

本方案将 trajectory 分支拆为 4 个层次：

1. **状态提取层**
   - 从历史 mask 序列提取低维轨迹状态
2. **轨迹编码层**
   - 将低维状态序列编码成条件向量
3. **SEG 调制层**
   - 用轨迹向量调制 `[SEG]` embedding
4. **辅助监督层**
   - 对轨迹表示增加轻量预测 / 对齐约束

---

## 3.1 轨迹状态定义

对每一帧的目标 mask，提取状态：

- `cx`: 目标中心 x
- `cy`: 目标中心 y
- `w`: bbox 宽
- `h`: bbox 高
- `area`: mask 面积
- `ratio`: `w / h`
- `vx`: 与前一帧中心位置差
- `vy`: 与前一帧中心位置差
- `darea`: 与前一帧面积差
- `q`: 质量/置信标记

建议统一为 10 维状态向量：

```text
s_t = [cx, cy, w, h, area, ratio, vx, vy, darea, q]
```

其中：

- 所有空间量按图像尺寸归一化到 `[0, 1]` 或 `[-1, 1]`
- `q` 表示该状态是否可靠（训练时一般取 1；推理时来自预测稳定性）

---

## 3.2 历史窗口定义

设历史窗口长度：

- `traj_history_len = 6`

对每个 question frame，我们希望得到其之前最多 6 帧的状态序列：

```text
H_t = [s_{t-6}, s_{t-5}, ..., s_{t-1}]
```

若不足 6 帧：

- 使用零填充
- 同时提供 `traj_valid_mask`

所以数据层输出应包括：

- `traj_states`: `[num_expr, question_frame_num, T_h, state_dim]`
- `traj_valid_mask`: `[num_expr, question_frame_num, T_h]`
- `traj_target_states`: `[num_expr, question_frame_num, state_dim]`

注意：
- `traj_target_states` 是当前 question frame 对应目标的真实状态，可用于辅助预测损失
- 这与当前 question frame 的 GT mask 一一对应

---

# 4. 模块设计

---

## 4.1 新增文件：`utils/trajectory_utils.py`

该文件只放**纯 torch / numpy 的无模型工具函数**，便于单元测试。

建议包含：

### 4.1.1 从 mask 提取状态

```python
mask_to_state(mask, eps=1e-6) -> Tensor[state_dim]
```

输入：
- `mask`: `[H, W]`，二值 mask

输出：
- `[cx, cy, w, h, area, ratio, vx, vy, darea, q]` 中的静态部分
- `vx, vy, darea` 可先置 0，由序列函数补

---

### 4.1.2 从 mask 序列提取状态序列

```python
masks_to_state_sequence(masks, image_h, image_w) -> Tensor[T, state_dim]
```

输入：
- `[T, H, W]`

输出：
- `[T, state_dim]`

内部负责：
- 计算 `cx, cy, w, h, area, ratio`
- 差分生成 `vx, vy, darea`
- 为空 mask 置合理默认值并设置 `q=0`

---

### 4.1.3 构建历史窗口

```python
build_history_windows(states, history_len) -> (hist, valid_mask, target)
```

输入：
- `states`: `[Q, state_dim]`，对应连续 `question_frame_num` 帧状态

输出：
- `hist`: `[Q, T_h, state_dim]`
- `valid_mask`: `[Q, T_h]`
- `target`: `[Q, state_dim]`

注意：
- 第一版建议**历史窗口仅基于 question frames 自身之前的帧**，不强依赖 context frames 的 GT mask
- 这样对现有数据结构侵入最小

---

### 4.1.4 推理时从预测 mask 更新状态窗口

```python
append_pred_state(history_states, history_valid, pred_mask, image_h, image_w)
```

用于推理时滑动更新。

---

## 4.2 新增文件：`model/trajectory_module.py`

该模块只负责“状态序列 -> trajectory 条件向量”。

建议核心类：

```python
class TrajectoryEncoder(nn.Module):
    ...
```

---

### 4.2.1 输入输出

输入：

- `traj_states`: `[N, T_h, state_dim]`
- `traj_valid_mask`: `[N, T_h]`

输出：

- `traj_global`: `[N, D]`
- `traj_tokens`: `[N, K, D]`（可选，先不一定用）
- `traj_pred`: `[N, state_dim]`（用于辅助预测）

其中：
- `N` 是“有效目标实例数”，通常可展平成 `batch × expression × question_frame`
- `D` 可以设为 `out_dim`，与 `[SEG]` embedding 维度对齐
- `K` 先保留接口，例如 `K=4`，但第一版可以不用到 token 注入

---

### 4.2.2 编码器结构建议

优先简单稳健：

```text
state sequence
  -> input MLP
  -> 2-layer temporal encoder
      - 优先 GRU
      - 若 mamba_ssm 可用，再提供 Mamba 版本可选
  -> masked attention pooling
  -> concat(last_valid_hidden, pooled_hidden)
  -> projection to out_dim
```

建议第一版默认：

- **双层 GRU**
- 把 Mamba 作为可选项，不作为主依赖

理由：
- 当前仓库无 `mamba_ssm` 依赖
- 先保证工程可跑
- 轨迹分支不应把 repo 依赖复杂度大幅抬高

---

### 4.2.3 调制头

增加两个小模块：

#### A. SEG 调制器
```python
traj_to_seg = MLP(D -> D)
```

输出 `delta_seg`

#### B. forecast head
```python
forecast_head = MLP(D -> state_dim)
```

用于预测当前 question frame 的目标状态

#### C. 可学习门控
```python
alpha = nn.Parameter(torch.tensor(0.0))
beta = nn.Parameter(torch.tensor(0.0))
```

解释：
- `alpha`: 控制 trajectory 对 `[SEG]` embedding 的调制强度
- `beta`: 控制 trajectory 对辅助对齐项的强度

初始化为 0 的好处是：
- 刚开始训练时几乎不影响原模型
- 训练可渐进启用 trajectory 信息

---

# 5. 数据层设计

---

## 5.1 为什么第一版不采样额外历史帧

表面上“在 question window 前再采 6 帧历史帧”很直观，但不建议第一版这么做，原因：

1. 当前数据集类已经有：
   - `context_frame_num`
   - `question_frame_num`
2. `masks` 目前只为 question frames 读取
3. 各数据集 mask 读取方式不同，额外采历史帧会显著增大实现复杂度

因此第一版建议：

> **历史状态仅从 question frames 的 GT mask 序列内部构造。**

也就是对 question 序列中的第 `j` 帧，其历史只来自：

- 同一个样本、同一个 expression、当前 question 窗口中前面的帧

例如：
- question 帧共 4 帧
- history_len 设为 6
- 实际有效历史最多只有前 0/1/2/3 帧，剩下 pad

优点：
- 不改现有图像采样逻辑
- 不额外读取更多 mask
- 可以先验证“显式轨迹状态是否有用”

第二版若有效，再考虑引入 question window 之外的历史帧。

---

## 5.2 修改 `dataset/refer_video_seg_dataset.py`

在当前返回值中新增：

- `traj_states`
- `traj_valid_mask`
- `traj_target_states`

构造方式：

1. 现有 `masks` 形状约为 `[num_expr, question_frame_num, H, W]`
2. 对每个 expression：
   - 取其 question mask 序列
   - 调 `masks_to_state_sequence`
   - 再调 `build_history_windows`

得到：

- `traj_states`: `[num_expr, question_frame_num, T_h, state_dim]`
- `traj_valid_mask`: `[num_expr, question_frame_num, T_h]`
- `traj_target_states`: `[num_expr, question_frame_num, state_dim]`

然后加入样本返回 tuple。

---

## 5.3 修改 `dataset/dataset.py`

### `HybridDataset.__getitem__`
保持逻辑不变，只透传新增字段。

### `collate_fn`
新增对 trajectory 字段的收集与打包：

输出增加：

- `traj_states_list`
- `traj_valid_mask_list`
- `traj_target_states_list`

保持风格与现有：

- `masks_list`
- `label_list`
- `resize_list`

一致，即 trajectory 数据先按 sample 保持 list，不强行 pad 成统一大张量，减少对多 expression 样本结构的破坏。

建议输出形式：

```python
"traj_states_list": List[Tensor[num_expr, Q, T_h, state_dim]]
"traj_valid_mask_list": List[Tensor[num_expr, Q, T_h]]
"traj_target_states_list": List[Tensor[num_expr, Q, state_dim]]
```

这样更贴合当前 `masks_list` 的处理方式。

---

# 6. 模型接入设计

---

## 6.1 修改 `GlusMetaModel`

文件：

- `model/GLUS.py`

在 `GlusMetaModel.initialize_glus_modules()` 中，在现有：

- `self.visual_model`
- `self.text_hidden_fcs`

之外，按需初始化：

```python
self.trajectory_encoder = TrajectoryEncoder(...)
```

仅在 `config.use_trajectory` 时创建；否则置为 `None`。

新增配置项：

- `use_trajectory`
- `traj_history_len`
- `traj_state_dim`
- `traj_hidden_dim`
- `traj_aux_loss_weight`
- `traj_align_loss_weight`

---

## 6.2 trajectory 不应注入 token embedding 层

第一版明确：

- **不在 `embed_tokens` 上挂 hook**
- **不引入 `[TRAJ_i]` token**
- **不改 `seg_token_mask` 构造逻辑**

trajectory 只在这里介入：

```text
LLM hidden states
  -> text_hidden_fcs
  -> 提取 [SEG] embeddings
  -> 用 trajectory encoder 输出调制 pred_embeddings
  -> SAM2 prompt encoder
```

这最贴合现有代码。

---

## 6.3 在 `generate_masks()` 中接入 trajectory

当前 `generate_masks()` 关键步骤：

1. 跑 LLM forward
2. 提取 `[SEG]` embedding
3. 按 `offset` 切分回每个 batch sample
4. 对每个 question frame 调用 SAM2 解码

trajectory 接入时机：

### 在 `pred_embeddings` 切分后、SAM2 解码前

因为这时已经恢复到了“按样本组织”的结构，更便于和：

- `traj_states_list`
- `traj_valid_mask_list`
- `traj_target_states_list`

对齐。

---

## 6.4 对齐方式

当前每个 sample：

- `pred_embeddings[i]` 对应一个样本内所有 expressions × question frames 的 `[SEG]` embedding 序列

并且现有代码已经假设其顺序为：

```text
expr0_frame0, expr0_frame1, ..., expr0_frameQ-1,
expr1_frame0, expr1_frame1, ..., expr1_frameQ-1,
...
```

这是由：

```python
cur_index = [j + question_frame_num * num for num in range(pred_embeddings[i].shape[0] // question_frame_num)]
```

可看出的。

因此 trajectory 也必须按同样顺序展平：

```python
traj_states_i: [num_expr, Q, T_h, state_dim]
-> flatten -> [num_expr * Q, T_h, state_dim]

traj_valid_i: [num_expr, Q, T_h]
-> flatten -> [num_expr * Q, T_h]

traj_target_i: [num_expr, Q, state_dim]
-> flatten -> [num_expr * Q, state_dim]
```

然后可与 `pred_embeddings[i]` 对齐。

---

## 6.5 `[SEG]` embedding 调制公式

设：

- 原始 `[SEG]` embedding：`e_seg`
- trajectory 编码器输出：`z_traj`
- 调制向量：`delta = traj_to_seg(z_traj)`

则第一版建议使用最简单稳定形式：

```text
e_seg' = e_seg + tanh(alpha) * delta
```

其中：
- `alpha` 是可学习标量，初始化 0

优点：
- 当训练初期或 trajectory 无效时，几乎不影响原路径
- 不改变 embedding 维度
- 不改 SAM2 prompt encoder 接口

若担心幅度问题，也可以使用：

```text
e_seg' = LayerNorm(e_seg + tanh(alpha) * delta)
```

但第一版建议尽量少加层。

---

## 6.6 trajectory 辅助损失

在 `generate_masks()` 中新增两个可选辅助项。

### 6.6.1 状态预测损失

trajectory encoder 输出：

- `traj_pred`: `[num_expr * Q, state_dim]`

监督目标：

- `traj_target_states`

损失：

```text
traj_pred_loss = SmoothL1(traj_pred, traj_target)
```

只对有效目标计算。

---

### 6.6.2 SEG-trajectory 对齐损失

目的：
- 让 `[SEG]` embedding 的几何信息与 trajectory 条件更一致

做法建议简单化：

- 先对 `e_seg` 和 `z_traj` 做 L2 normalize
- 做 cosine 对齐

例如：

```text
traj_align_loss = 1 - cosine(e_seg_detached_or_proj, z_traj)
```

更稳妥版本建议：
- 给 `[SEG]` embedding 再过一个小投影层到同维空间
- 只做弱约束，权重较小

注意：
- 第一版可只开 `traj_pred_loss`
- `traj_align_loss` 作为次级可选项，避免引入不稳定性

---

## 6.7 总损失

在原始：

```text
loss = ce_loss + mask_loss (+ contrastive_loss)
```

基础上，若 `use_trajectory`：

```text
loss = loss
     + traj_aux_loss_weight * traj_pred_loss
     + traj_align_loss_weight * traj_align_loss
```

这里要强调：

- 不是“原始损失不变”
- 而是“原始损失项保留，trajectory 只做额外加项”

---

## 6.8 `evaluate()` 中接入 trajectory

推理路径 `evaluate()` 与训练不同：

- 它使用 `self.generate(...)`
- 然后提取生成序列里的 `[SEG]`

因此 trajectory 接入仍应选在：

- `[SEG]` embedding 已经提取出来之后
- 送入 SAM2 前

即与训练路径保持同一位置。

---

# 7. 推理阶段 trajectory 设计

---

## 7.1 推理不使用 GT，只能依赖预测 mask

在 `inference_iter.py -> inference_frames()` 中，每一步 decode 后都能得到：

- 当前预测 mask

因此可以维护一个 per-expression 的状态滑窗。

但当前推理实现主要是：

- 一次只处理一个 expression
- batch size 基本为 1
- 对话滚动推进

这反而使 trajectory 推理维护更简单。

---

## 7.2 推理侧状态缓存结构

建议在 `inference_frames()` 中维护：

```python
traj_history_masks = deque(maxlen=traj_history_len)
traj_history_states = deque(maxlen=traj_history_len)
```

每次当前帧预测完成后：

1. 从预测二值 mask 提取当前状态
2. 根据上一步状态补 `vx, vy, darea`
3. 追加到滑窗
4. 下一步构造：
   - `traj_states`: `[1, curr_q_num_or_1, T_h, state_dim]`
   - 或更简单：当前只给“最后一步待解码的 SEG”提供 `[1, T_h, state_dim]`

---

## 7.3 推理第一版建议：只调制“当前最后一步 SEG”

由于 `decode_iter=True` 场景下，真正最终取用的是：

- 当前迭代最后一步生成的 mask

因此第一版推理不必强求对所有历史 question positions 都构造 trajectory，只需：

> 对当前最终用于 decode 的那个 `[SEG]` embedding 做 trajectory 调制。

这样能显著降低实现复杂度。

---

## 7.4 置信度/质量 `q`

训练时：
- `q=1`（若 GT mask 非空）
- 空 mask 时可置 0

推理时：
- `q` 应来自预测 mask 可靠性
- 第一版建议简单使用：
  - 若 mask 非空，则 `q=1`
  - 若 mask 为空，则 `q=0`
- 后续增强版再考虑结合：
  - `iou_predictions`
  - mask 面积变化稳定性
  - 连续帧漂移程度

不要在第一版就加入复杂“置信度门控技巧”。

---

# 8. 参数设计

---

## 8.1 `train_ds.py` 新增 CLI

建议新增：

- `--use_trajectory`
- `--traj_history_len`，默认 `6`
- `--traj_state_dim`，默认 `10`
- `--traj_hidden_dim`，默认 `256`
- `--traj_aux_loss_weight`，默认 `0.1`
- `--traj_align_loss_weight`，默认 `0.0`
- `--traj_encoder_type`，默认 `gru`，可选 `gru|mamba`

并将它们加入 `model_args`。

---

## 8.2 `inference_iter.py` 新增 CLI

建议新增：

- `--use_trajectory`
- `--traj_history_len`
- `--traj_state_dim`
- `--traj_encoder_type`

注意：
- 推理端不需要辅助损失权重参数
- 但需要与训练时的 trajectory 结构参数对齐

---

# 9. 文件级修改清单

---

## 9.1 新增文件

### `utils/trajectory_utils.py`
用途：
- mask -> state
- state sequence -> history windows
- padding / valid mask
- 推理侧状态滑窗辅助

### `model/trajectory_module.py`
用途：
- `TrajectoryEncoder`

### `scripts/train_glus_traj.sh`
用途：
- 提供 trajectory 训练脚本

### `scripts/inference_traj.sh`
用途：
- 提供 trajectory 推理脚本

### `trajectory_experiment_notes.md`
用途：
- 实验记录模板
- 可复现性说明

---

## 9.2 修改文件

### `dataset/refer_video_seg_dataset.py`
新增：
- 构造 `traj_states`, `traj_valid_mask`, `traj_target_states`

### `dataset/dataset.py`
新增：
- collate trajectory 相关字段

### `model/GLUS.py`
新增：
- trajectory encoder 初始化
- `generate_masks()` 中 trajectory 调制与辅助 loss
- `evaluate()` 中 trajectory 调制

### `train_ds.py`
新增：
- trajectory CLI
- 将参数传给模型

### `inference_iter.py`
新增：
- trajectory CLI
- 维护预测 mask 状态滑窗
- 调 `evaluate()` 时传 trajectory 输入

---

# 10. 推荐实施顺序

---

## Phase 1：最小可行版本（推荐先做）

目标：
- 不改 tokenizer
- 不改 prompt
- 不加 `[TRAJ_i]`
- 不加 hook
- 只在 `[SEG]` embedding 后做 trajectory 调制

具体：
1. `utils/trajectory_utils.py`
2. `model/trajectory_module.py`
3. `dataset/refer_video_seg_dataset.py`
4. `dataset/dataset.py`
5. `model/GLUS.py`
6. `train_ds.py`
7. 写一个纯 torch smoke test，测试：
   - mask->state
   - history window
   - encoder forward
   - 调制输出 shape 对齐

预期：
- 可以稳定训练
- 便于验证 trajectory 信息是否有增益

---

## Phase 2：推理接入

在 Phase 1 训练通后，再改：

- `inference_iter.py`
- 滑窗状态维护
- 当前步 `[SEG]` trajectory 调制

先不要做复杂双向置信门控。

---

## Phase 3：增强版（仅在前两阶段有效后考虑）

若 trajectory 分支确实带来收益，再考虑：

1. question window 外历史帧采样
2. `[TRAJ_i]` token 显式注入
3. token-level conditioning
4. Mamba 编码器替代 GRU
5. 更复杂的 trajectory confidence gating

---

# 11. 风险点与规避策略

---

## 11.1 风险：trajectory 与 `pred_embeddings` 顺序错位
规避：
- 明确使用当前已有的 expression-major、frame-minor 展平顺序
- 在数据层与模型层都写 shape 断言

---

## 11.2 风险：trajectory 辅助项破坏原模型稳定性
规避：
- `alpha` 初始化为 0
- `traj_aux_loss_weight` 从小值开始
- 第一版先不开 `traj_align_loss`

---

## 11.3 风险：推理误差积累
规避：
- 第一版只用最近历史预测 mask
- `q` 简化为有效/无效标记
- 暂不引入复杂置信度传播机制

---

## 11.4 风险：多数据集兼容性
规避：
- 第一版只从现有 `masks` 计算 trajectory 状态
- 不额外新增历史帧 mask 读取逻辑
- 避免触碰各 loader 的底层结构

---

# 12. 最终结论

本设计相对于“引入 `[TRAJ_1]...[TRAJ_4]` token 并 hook embedding”方案，更贴合 GLUS 当前仓库的真实结构，因为它：

1. **遵循 GLUS 当前核心主链**
   - LLM -> `[SEG]` hidden state -> SAM2

2. **不破坏现有 prompt / token / offset 机制**
   - 避免重写最脆弱的对齐逻辑

3. **只在最合适的接口层插入 trajectory 条件**
   - 即 `[SEG]` embedding 抽取之后

4. **先验证 trajectory 信息本身是否有效**
   - 再决定是否升级为更复杂的 token-level 注入方案

因此，这是一版更现实、可分阶段推进、且与当前仓库结构兼容性更高的 trajectory 技术设计。

---

# 13. 建议的最小实现口径（一句话版）

> 第一版 trajectory 分支不改 GLUS 的文本 token 机制，只从 question window 的 GT / 预测 mask 序列中提取低维运动状态，用一个轻量时序编码器把历史状态编码成条件向量，并在 `[SEG]` embedding 送入 SAM2 前对其做残差调制，同时增加一个轻量状态预测辅助损失；所有行为由 `--use_trajectory` 控制。
