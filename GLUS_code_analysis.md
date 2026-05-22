GLUS 仓库代码分析：核心实现与数据流梳理
仓库：GLUS-video/GLUS
论文描述：[CVPR 2025] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation

1. 仓库整体结构概览
从仓库根目录看，GLUS 的主线代码主要集中在以下位置：

train_ds.py
主训练入口
负责参数解析、模型构建、LoRA 注入、DeepSpeed 初始化、训练/验证循环
inference_iter.py
主推理入口
负责加载模型、逐视频表达式推理、迭代式生成分割结果并保存 mask
model/GLUS.py
GLUS 核心实现
定义 GLUSForCausalLM、视觉编码与 memory bank、文本 token 到 mask 的映射、训练 loss 和推理逻辑
dataset/dataset.py
统一数据集包装
定义 HybridDataset 和 collate_fn
dataset/refer_video_seg_dataset.py
具体的视频指代表达分割数据读取逻辑
负责采样 context/question 帧、构造对话 prompt、生成 GT mask
dataset/*.py
各具体数据集的元信息加载器
如 mevis.py、revos.py、refyoutube_vos.py、davis17.py、lvvis.py
utils/utils.py
prompt 模板、训练统计、IoU 工具、CUDA 搬运等
utils/contrastive_loss.py
可选对比学习损失
scripts/train_glus_a.sh, scripts/train_glus_s.sh
训练启动脚本
scripts/inference.sh
推理启动脚本
demo.ipynb
notebook 形式的演示入口，作用偏 demo / 使用示例，不是主训练主推理实现核心
model/llava/
LLaVA/LLaMA 多模态语言模型基础代码
model/segment-anything-2/
引入的 SAM2 相关代码
GLUS 在其上做文本引导的视频分割与 memory bank 组织
2. 主线入口总结
2.1 训练入口
训练主入口是：

train_ds.py
训练脚本调用示例见：

scripts/train_glus_a.sh
scripts/train_glus_s.sh
典型启动链路：

scripts/train_glus_a.sh
  -> deepspeed train_ds.py
     -> parse_args()
     -> 构造 tokenizer
     -> GLUSForCausalLM.from_pretrained(...)
     -> 初始化 vision tower + GLUS 模块
     -> 构造 HybridDataset / DataLoader(由 deepspeed.initialize 接管)
     -> train(...)
        -> model(**input_dict)
           -> GLUSForCausalLM.forward()
              -> generate_masks()
                 -> 语言输出 hidden states
                 -> 提取 [SEG] 对应 embedding
                 -> SAM2 prompt encoder / mask decoder
                 -> 计算 CE + BCE + Dice (+ contrastive) loss
2.2 推理入口
推理主入口是：

inference_iter.py
脚本示例：

scripts/inference.sh
典型链路：

scripts/inference.sh
  -> python inference_iter.py
     -> parse_args()
     -> tokenizer / model 加载
     -> inference_video(...)
        -> get_frame_paths_and_target_obj(...)
        -> inference_frames(...)
           -> 构造逐帧对话
           -> model.evaluate(...)
              -> LLM generate
              -> 提取 [SEG] token hidden states
              -> SAM2 decode mask
              -> 使用 memory bank 做时序传播
        -> 保存 png mask
2.3 Notebook 的角色
demo.ipynb 体积很大，明显更偏演示/交互使用。
从工程主线看，真正的训练和推理主逻辑仍在 .py 脚本中，尤其是 train_ds.py 和 inference_iter.py。
所以 notebook 更适合：
快速跑 demo
观察中间结果
做可视化展示
不适合作为理解主实现时的唯一依据。
3. GLUS 的核心设计：整体理解
GLUS 本质上是在一个 LLaVA/LLaMA 风格的大语言模型 上，叠加了：

语言-视觉多模态理解能力
来自 LLaVA 风格的 vision tower + LLM
文本到分割 mask 的映射
使用特殊 token [SEG]
从 LLM hidden states 中取出 [SEG] 对应向量，映射到 segmentation prompt embedding
SAM2 视觉分割解码器
用文本 embedding 作为 prompt 输入 sam_prompt_encoder
再由 sam_mask_decoder 输出 mask
视频时序 memory bank
利用前序帧的 mask memory 引导当前帧
这是“global-local reasoning”里非常关键的一部分：不仅看当前帧，还维护跨帧记忆
对话式逐帧推理
把视频分割组织成多轮对话
前几帧作为 context frames，后续 question frames 逐步问“请分割该目标”
由 LLM 输出包含 [SEG] 的回答，再转为 mask
4. 核心实现文件与类
4.1 核心模型类
文件：

model/GLUS.py
关键类：

GlusMetaModel
GlusModel
GLUSForCausalLM
其中最重要的是：

GLUSForCausalLM
4.1.1 GlusMetaModel
职责：

初始化 GLUS 额外模块
建立 SAM2 视觉模型
建立文本 hidden state 到分割 embedding 的投影层
关键代码逻辑：

initialize_glus_modules(self, config)
它做了两件关键事：

A. 构建 SAM2
self.visual_model = build_sam2(...)
说明 GLUS 的 mask 预测底座是 SAM2。

B. 文本投影层
self.text_hidden_fcs = nn.ModuleList([nn.Sequential(...)])
输入是 LLM hidden state，输出是 out_dim 维分割提示向量。
这相当于把语言空间表示投影到可供 SAM prompt encoder 使用的空间。

4.1.2 GlusModel
GlusModel(GlusMetaModel, LlavaLlamaModel) 继承了：

GLUS 额外模块
LLaVA/LLaMA 主体
它主要负责把 LLaVA 模型配置和 GLUS 配置整合到一起。

4.1.3 GLUSForCausalLM
这是最核心的封装类，负责：

训练 forward
推理 evaluate
memory bank 管理
segmentation token 提取
mask 解码
loss 计算
关键成员包括：

self.seg_token_idx
self.memory_bank_list
self.num_maskmem
self.transform = SAM2Transforms(...)
self.contrastive_loss
5. GLUS 如何把 LLM 与视频分割结合起来
5.1 特殊分割 token [SEG]
训练时，在 train_ds.py 中：

tokenizer 添加 token：[SEG]
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
作用：

LLM 输出文本中出现 [SEG]
模型从 hidden states 中定位这些 [SEG] 的位置
将这些位置的 hidden states 作为 segmentation prompt embedding
这一步是整个系统的桥梁：
语言 token -> 分割提示向量 -> SAM mask

5.2 文本隐藏状态转分割 embedding
在 model/GLUS.py 的 generate_masks() 与 evaluate() 中：

先运行 LLM，得到 output_hidden_states
用 self.model.text_hidden_fcs[0](...) 做投影
通过 seg_token_mask 取出 [SEG] 对应位置的 embedding
伪代码如下：

output = super().forward(..., output_hidden_states=True)
hidden = output.hidden_states[-1]
proj_hidden = text_hidden_fc(hidden)
pred_embeddings = proj_hidden[seg_token_mask]
这说明：

[SEG] 不是直接输出 mask
而是先变成一个隐向量提示
再喂给 SAM2 的 prompt encoder
5.3 用 SAM2 做 mask 解码
在 generate_masks() / evaluate() 中，每个 question frame 都会执行：

sparse_embeddings, dense_embeddings = self.model.visual_model.sam_prompt_encoder(
    points=None,
    boxes=None,
    masks=None,
    text_embeds=pred_embeddings[i][cur_index].unsqueeze(1),
)
然后送入：

low_res_masks, iou_predictions, sam_output_tokens, object_score_logits = \
    self.model.visual_model.sam_mask_decoder(...)
再通过：

pred_mask = self.transform.postprocess_masks(low_res_masks, orig_hw=...)
恢复到原图尺寸。

这说明 GLUS 的分割逻辑是：

LLM 负责理解目标表达 / 上下文推理
SAM2 负责掩码解码
两者靠 [SEG] 对应的 embedding 连接
6. Global-Local Reasoning 是如何组织的
从代码来看，GLUS 的 “Global-Local” 不是一个单独命名的模块，而是体现在输入构造 + 时序记忆 + 对话迭代的整体设计里。

可以把它拆成三层：

6.1 Global：全局上下文帧
在数据集与推理中，都会选取：

context_frame_num 个上下文帧
例如默认是 4 帧。

这些 context frame 通过 CONTEXT_INFO_LIST 被放到 prompt 前缀里：

DEFAULT_IMAGE_TOKEN * CONTEXT_FRAME_NUM + "\nPlease finished the following tasks on the context frames above.\n"
即先给模型若干帧全局上下文，让它理解视频整体目标。

6.2 Local：当前待分割帧
然后每个 question frame 对应一个问题模板，例如：

Can you segment the {class_name} in this frame?
也就是局部单帧上的定位与分割。

6.3 Unified：通过单个 LLM 对话序列统一组织
训练数据不是把“全局视频理解”和“局部单帧分割”拆成两个模型，而是统一到一个多轮对话中：

上文先提供 context frames
再连续问多个 question frames
每个回答含 [SEG]
每个 [SEG] 对应一帧或一个表达实例的 mask
所以“Unified into A Single Large Language Model”在代码层面的体现就是：

所有 reasoning 由同一个 LLaVA/LLaMA 序列统一处理
再把语言隐状态投影到分割空间
7. Memory Bank：视频时序信息的关键实现
这是 GLUS 区别于纯图像级 LLM+SAM 分割的重要部分。

7.1 memory bank 存储结构
在 GLUSForCausalLM.__init__：

self.memory_bank_list = {} 或 None
self.num_maskmem = 7
memory bank 按 frame_id 存储历史信息。

7.2 当前帧如何读取历史记忆
在 get_visual_embs() 中：

若没有历史 memory，直接给当前视觉特征加 no_mem_embed
若有历史 memory，则遍历过去若干帧，从 memory_bank_list 中取出 memory features 和位置编码
再调用：
self.model.visual_model.memory_attention(
    curr=[vision_feats[-1][:, bs_l:bs_r]],
    curr_pos=[vision_pos_embeds[-1][:, bs_l:bs_r]],
    memory=memory,
    memory_pos=memory_pos,
    num_obj_ptr_tokens=0,
)
这表示：

当前帧高层视觉特征会和历史帧 memory 做 attention
从而实现跨帧传播 / 时序增强
7.3 当前帧推理后如何写回 memory
在 generate_masks() 和 evaluate() 中，mask 解码后调用：

maskmem_features, maskmem_pos_enc = self.model.visual_model._encode_new_memory(
    current_vision_feats=vision_feats_curr_batch,
    feat_sizes=self.feat_sizes,
    pred_masks_high_res=...,
    is_mask_from_pts=False,
)
然后写入：

self.memory_bank_list[rel_pos_list[j]].append([
    maskmem_features.detach(),
    maskmem_pos_enc[0].detach(),
])
所以 memory 的来源是：

当前帧视觉特征
当前预测 mask
这符合视频分割常见范式：
上一帧/历史帧的分割结果反哺当前帧。

8. 训练数据流：端到端链路
下面按训练过程梳理完整数据流。

8.1 数据集初始化
入口：

train_ds.py
构造训练集：

train_dataset = HybridDataset(...)
HybridDataset 定义在：

dataset/dataset.py
它内部当前主用：

ReferVideoSegDataset(...)
也就是实际上训练主要围绕视频指代表达分割数据。

8.2 各数据集元信息加载
ReferVideoSegDataset.__init__ 中按名字加载：

load_mevis_json
load_revos_json
load_lvvis_json
load_refyoutube_json
load_davis17_json
分别位于：

dataset/mevis.py
dataset/revos.py
dataset/lvvis.py
dataset/refyoutube_vos.py
dataset/davis17.py
这些 loader 负责把每个视频表达组织成类似 meta 字典，包含：

video
exp
obj_id
anno_id
frames
length
file_names
exp_id
str_id
8.3 样本采样
在 ReferVideoSegDataset.__getitem__ 中：

第一步：选择数据源
按 sample_rate 从多个数据集里选一个：

mevis
revos
lvvis
refyoutube_vos
davis17
第二步：选择视频
随机抽一个 video。

第三步：选择表达
从该视频的多个表达里，采样最多 num_classes_per_sample 个表达。

默认：

num_classes_per_sample = 3
所以一个样本可能包含一个视频里的多个目标表达。

8.4 帧采样
关键参数：

context_frame_num
question_frame_num
默认通常都是 4。

采样方式：

question frames
连续采样一段：

sampled_start = random.randint(0, video_length - self.question_frame_num)
sampled_question_indices = list(range(sampled_start, sampled_start + self.question_frame_num))
context frames
把整段视频均匀切成若干区间，每段随机取一帧：

indices = np.linspace(0, video_length, self.context_frame_num + 1, dtype=int)
sampled_context_indices = [np.random.choice(range(indices[i], indices[i + 1])) for i in range(self.context_frame_num)]
最终：

sampled_indices = sampled_context_indices + sampled_question_indices
所以输入帧序列长度为：

context_frame_num + question_frame_num
默认就是 8 帧。

8.5 图像预处理
对每帧做两路处理：

路 1：给 CLIP / LLaVA 用
image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
路 2：给 SAM2 用
image = self.transform(image).contiguous()
所以每一帧会生成两种视觉输入：

image_clip: 供 LLaVA vision tower 用
image: 供 SAM2 / segmentation 流程用
8.6 对话构造
ReferVideoSegDataset.__getitem__ 中，为每个 sampled expression 构造多轮对话。

每个表达会构造 question_frame_num 个问题，例如：

第一轮带上下文提示 context_info
后续轮只问当前帧
回答模板来自：

utils/utils.py 中的 ANSWER_LIST
例如：

"Sure, the segmentation result is [SEG]."
这意味着训练时监督语言模型学会在回答里产生 [SEG]。

8.7 GT mask 读取
对每个 sampled expression，会读取所有 question frames 的 GT mask。

不同数据集处理略有区别：

mevis / revos / lvvis：通过 mask_dict 读 RLE
refyoutube_vos / davis17：通过 mask PNG 读 label map
最终得到：

masks: shape 近似为 [num_expr, question_frame_num, H, W]
代码可见：

masks = np.stack(masks, axis=0)
masks = torch.from_numpy(masks)
8.8 batch 组装与 collate
在 dataset/dataset.py 的 collate_fn：

收集并输出：

images: [B, T, C, H, W]
images_clip: [B, T, C, Hc, Wc]
input_ids
labels
attention_masks
masks_list
label_list
resize_list
offset
sampled_str_ids_list
sampled_frames_list
offset 的作用
由于一个 batch 中每个样本可能包含多个 conversations（每个表达一个 conversation），
offset 用来记录“哪个 conversation 属于哪个视频样本”。

这是后面把语言输出重新映射回每个样本的关键。

8.9 训练前向过程
在 train_ds.py -> train()：

output_dict = model(**input_dict)
进入：

GLUSForCausalLM.forward()
若非 past_key_values 场景，则转到 generate_masks()
8.10 generate_masks() 内部数据流
这是训练核心。

阶段 A：构造 seg_token_mask
代码通过 input_ids == seg_token_idx 找到 [SEG] 位置。
但这里并不是简单直接取，而是考虑了：

前部 context frame 对应的 image features 占位
question frame 的 image feature 插槽
说明模型把图像 token 和文本 token 混合在同一序列逻辑里，需要重新定位 [SEG] 对应位置。

阶段 B：跑 LLM 主干
output = super().forward(
    images=images_clip,
    attention_mask=attention_masks,
    input_ids=input_ids,
    labels=labels,
    output_hidden_states=True,
    specific_ce_loss=True,
)
这里的 images_clip 是按 conversation 展开的。

输出得到：

output.hidden_states
output.loss（语言 CE loss）
阶段 C：提取 [SEG] embedding
last_hidden_state = text_hidden_fcs(output_hidden_states[-1])
pred_embeddings = last_hidden_state[seg_token_mask]
然后结合 offset 和每个样本的 [SEG] 个数，把 embedding 切回每个 batch sample。

阶段 D：逐 question frame 解码 mask
对 j in range(question_frame_num)：

调 get_visual_embs(images[:, j + context_frame_num], ...)
编码当前 question frame 的视觉特征
若有 memory bank，则融合历史 memory
根据当前 frame 对应的 [SEG] embedding 构造 text_embeds
输入 SAM2 prompt encoder
输入 SAM2 mask decoder
上采样得到当前帧 mask
输出组织成：

pred_masks[frame_id][batch_id]
阶段 E：写入 memory bank
如果启用 memory bank，就把本帧的预测 mask 编码成新的 memory，供后续帧使用。

阶段 F：计算 loss
损失由三部分组成：

1. 语言损失
ce_loss = model_output.loss * ce_loss_weight
2. mask BCE loss
sigmoid_ce_loss(...)
3. mask Dice loss
dice_loss(...)
4. 可选 contrastive loss
utils/contrastive_loss.py
总损失：

loss = ce_loss + mask_loss
if contrastive_loss is not None:
    loss += contrastive_loss
9. 推理数据流：端到端链路
下面梳理 inference_iter.py 的推理路径。

9.1 模型加载
在 main()：

加载 tokenizer
获取 [SEG] token id
GLUSForCausalLM.from_pretrained(...)
初始化 vision modules
切换到 eval
这里支持：

bf16 / fp16 / fp32
4bit / 8bit 量化
deepspeed inference（fp16 场景）
9.2 选取待推理视频与表达
get_frame_paths_and_target_obj(args)：

按数据集读取 metas，并遍历：

视频 ID
表达 ID
找到当前还没生成结果的表达目录，就返回：

file_names
exp
dataset_settings
frames
这里推理是按表达逐个处理的，而不是一次性整个验证集 all-in-one。

9.3 视频级推理组织
在 inference_video()：

先取视频所有帧路径
取表达文本 target_obj
构造 context clips
做 forward inference
若启用了关键帧/双向策略，再做 backward inference
保存每一帧的 mask PNG
从代码看，它通过把视频拆成：

一组 context frames
后续待预测 frames
并采用迭代式 decode。

9.4 inference_frames() 逐帧迭代推理
这是推理关键函数。

预处理阶段
对所有输入帧，分别得到：

full_images：SAM2 路
full_images_clip：LLaVA 路
对话滚动构造
对每个时间步 i：

第一次问题会包含 context info
后续问题只问当前帧
如果超过 question_frame_num，就把最早一轮对话丢掉，形成滑动窗口
说明推理时是一个滚动对话状态机。

调用模型
output_ids, pred_masks, outputs = model.evaluate(...)
evaluate 内部逻辑
model/GLUS.py -> evaluate()：

先 generate(...) 让 LLM 输出文本
定位 [SEG] token
提取 hidden states 并映射成 pred_embeddings
调 get_visual_embs() 编码当前 question frame
送给 SAM2 prompt encoder + mask decoder
输出当前 mask
写入 memory bank
保存结果
最终二值化：

(pred_masks[-1][0] > 0).int()
并保存为 png。

10. 关键调用链总结
10.1 训练调用链

10.2 推理调用链

11. 关键张量/数据结构语义与大致形状
下面给出从代码中能直接推断出的主要张量语义。

11.1 数据集输出
单样本返回：

images
形状近似：[T, 3, 1024, 1024]
其中 T = context_frame_num + question_frame_num
image_clips
形状近似：[T, 3, Hc, Wc]
给 CLIP/LLaVA 的视觉塔
conversations
长度约等于表达个数 num_expr
masks
形状：[num_expr, question_frame_num, H, W]
sampled_str_ids
每个表达一个 string id
sampled_frames
question frames 对应帧名列表
11.2 collate 后
images
[B, T, 3, 1024, 1024]
images_clip
[B, T, 3, Hc, Wc]
input_ids
[N_conv, L]
labels
[N_conv, L]
attention_masks
[N_conv, L]
offset
[B+1]
用于从 conversation 维度映射回 batch 维度
这里 N_conv 不一定等于 B，因为每个样本可能有多个表达/多条 conversation。

11.3 LLM 输出
output_hidden_states[-1]
[N_conv, seq_len, hidden_size]
经过 text_hidden_fcs：

[N_conv, seq_len, out_dim]
通过 seg_token_mask 提取后：

pred_embeddings
总量约等于所有 conversation 中 [SEG] 数量总和
再被切分回每个样本
11.4 SAM2 视觉特征
get_visual_embs() 中：

image_embed
当前帧主特征图
high_res_feats
高分辨率辅助特征图列表
vision_feats
多尺度特征，供 memory encode / attention 使用
从代码里：

self.feat_sizes = [(256, 256), (128, 128), (64, 64)]
可推断多尺度特征大致对应这些空间尺寸。

11.5 mask 输出
SAM2 输出：

low_res_masks
经 postprocess_masks 后得到：
pred_mask
形状约为 [num_expr, 1, H, W]
最终常用：

pred_mask[:, 0]
[num_expr, H, W]
训练里，pred_masks 的组织方式是：

外层按 question_frame_num
内层按 batch sample
可理解成：

pred_masks[frame_idx][batch_idx] -> [num_expr, H, W]
12. 训练目标与 loss 接入方式
12.1 CE loss
来源：

LLM 的语言建模损失
监督方式：

collate_fn 会把用户指令部分 mask 成 IGNORE_INDEX
只监督 assistant 回复部分
assistant 回复里包含 [SEG]
因此 CE loss 在学的是：

按对话方式输出合理回答
并在需要时输出 [SEG]
12.2 分割损失
在 model/GLUS.py 中定义：

sigmoid_ce_loss
dice_loss
对每个样本、每个 question frame：

将预测 mask 与 GT mask 对齐
累积 BCE + Dice
最终：

mask_loss = mask_bce_loss + mask_dice_loss
12.3 对比损失
可选：

utils/contrastive_loss.py
它维护一个 seg_token_bank，核心思想是：

同一视频中指向相同对象的不同表达 / 帧，其 segmentation token 应更接近
不同对象或不同表达应分开
目前主要针对 mevis。

13. 配置与参数机制
13.1 参数解析入口
训练：

train_ds.py -> parse_args()
推理：

inference_iter.py -> parse_args()
13.2 关键超参数
训练相关
--epochs
--steps_per_epoch
--batch_size
--grad_accumulation_steps
--lr
--precision
模型相关
--version
--vision-tower
--sam_config
--vision_pretrained
--out_dim
--image_features_num
--not_use_mem_bank
--use_contrastive_loss
视频分割相关
--context_frame_num
--question_frame_num
--total_question_frame_num
损失相关
--ce_loss_weight
--dice_loss_weight
--bce_loss_weight
--contrastive_loss_weight
数据相关
--dataset_dir
--refer_video_seg_data
--sample_rates
13.3 实验切换方式
主要通过 shell 脚本和命令行参数切换：

scripts/train_glus_a.sh
scripts/train_glus_s.sh
scripts/inference.sh
例如训练脚本中可切换：

使用哪些数据集：
--refer_video_seg_data="mevis||refyoutube_vos||davis17||revos||lvvis"
采样比例：
--sample_rates "40,150,4,30,30"
context/question frame 数量：
--context_frame_num 4
--question_frame_num 4
值得注意的是脚本里专门写了注释：

修改 context_frame_num 和 question_frame_num 时，还要同步修改 utils.utils

这说明 prompt 模板中的 <image> 数量是写死依赖这些常量的。

14. 从代码视角总结 GLUS 的核心创新实现
基于代码实现，可以把 GLUS 的核心归纳为以下几点。

14.1 单一 LLM 统一全局与局部视频分割推理
GLUS 并没有拆成：

一个视频编码器做全局理解
一个图像分割器做局部预测
而是通过：

context frames + question frames
多轮对话
[SEG] token
把全局-局部推理统一放到一个多模态 LLM 序列建模里。

14.2 [SEG] 作为语言到分割的桥接点
这部分是实现上的核心桥梁：

语言模型输出 [SEG]
[SEG] 对应 hidden state
hidden state -> text_hidden_fcs -> segmentation embedding
segmentation embedding -> SAM2 prompt encoder
SAM2 decoder -> mask
即：

Language token -> hidden state -> segmentation prompt -> mask
14.3 时序 memory bank 让视频分割具备传播能力
通过：

_encode_new_memory(...)
memory_attention(...)
把前序帧的预测结果转成 memory，反馈给当前帧特征。
这使得 GLUS 不只是“逐帧问答 + 分割”，而是具备明确的视频时序建模能力。

14.4 对话滑动窗口式推理
在推理里，GLUS 不是一次性把整段视频全塞给模型，而是：

保留固定数量 context frames
逐步推进 question frames
动态更新对话历史
必要时丢掉最早轮次
这是一种适应 LLM 上下文窗口限制的工程实现。

15. 一份更抽象的数据流总图

16. 代码层面的几个实现细节与注意点
16.1 forward() 在训练时直接走 generate_masks()
GLUSForCausalLM.forward() 中：

若存在 past_key_values，走父类 forward
否则直接 return self.generate_masks(**kwargs)
说明普通训练不是标准 HuggingFace CausalLM forward 输出，而是 GLUS 自定义训练输出字典。

16.2 validate() 代码与当前 generate_masks() 返回值可能并不完全一致
在 train_ds.py -> validate() 里期待：

output_dict["pred_masks"]
output_dict["gt_masks"]
但当前 model/GLUS.py -> generate_masks() 返回的只有：

loss
ce_loss
mask_bce_loss
mask_dice_loss
mask_loss
contrastive_loss
所以从当前代码看，validation 路径可能不是完全对齐的，或者仓库此部分存在未同步修改。
而 dataset/dataset.py 里也写了注释：

ValDataset is not used in GLUS.

这说明主线训练更偏向：

训练
独立推理脚本做评测/生成
而不是强依赖 train_ds.py 内嵌 validation。

16.3 utils/utils.py 中 context/question 数量是常量绑定的
CONTEXT_FRAME_NUM=4
QUESTION_FRAME_NUM=4
prompt 模板直接使用这些常量展开 <image> 个数。
所以如果改帧数，不仅要改训练参数，还要改 prompt 模板文件。

16.4 训练样本的“一个 batch 样本”其实可能含多个表达
这点很重要：

一个视频样本内会采多个 expression
每个 expression 有自己的 conversation
但共享同一组视频帧
因此 GLUS 的 batch 语义不是“一个样本对应一个目标”，而是更接近：

一个视频 clip + 多个表达目标 + 多轮对话
17. 结论
从代码实现来看，GLUS 的核心并不是简单的 “LLaVA + SAM2 拼接”，而是一个较完整的统一式视频分割框架，特点包括：

以多轮对话的方式组织视频分割任务

context frames 提供全局上下文
question frames 执行局部分割
通过 [SEG] token 把 LLM 隐状态映射为分割提示

这是语言理解与 mask 解码的关键桥梁
以 SAM2 作为 mask 解码底座

prompt encoder 接收文本 embedding
mask decoder 负责掩码生成
通过 memory bank 实现时序传播

当前帧可读取历史 mask memory
当前预测又会写回 memory
训练目标联合了语言建模与分割监督

CE loss 学会输出对话和 [SEG]
BCE + Dice loss 学会生成准确 mask
可选 contrastive loss 强化 token 表达一致性
因此，若用一句话概括代码层面的 GLUS：

GLUS 用一个统一的多模态 LLM 处理“视频上下文理解 + 当前帧分割请求”，再把 [SEG] token 的语言隐向量投影为 SAM2 的文本分割提示，并通过 memory bank 将时序信息持续注入后续帧分割。

18. 附：最关键文件索引
核心模型
model/GLUS.py
训练入口
train_ds.py
推理入口
inference_iter.py
数据集包装
dataset/dataset.py
视频分割数据主实现
dataset/refer_video_seg_dataset.py
prompt / 常量 / 工具
utils/utils.py
对比损失
utils/contrastive_loss.py
训练脚本
scripts/train_glus_a.sh
scripts/train_glus_s.sh
推理脚本
scripts/inference.sh
Skip to content
GLUS-video
GLUS
Repository navigation
Code
Issues
1
 (1)
Pull requests
Agents
Actions
Projects
Security and quality
Insights
Owner avatar
GLUS
Public
GLUS-video/GLUS
Name		
author
Lang Lin
fixed a small bug: return an extra None when no <SEG> generated.
8fd138b
 · 
11 months ago
assets
add overview
last year
dataset
initial commit
last year
demo
fixed a demo bug
last year
kfs
merge
last year
model
fixed a small bug: return an extra None when no <SEG> generated.
11 months ago
scripts
add inference dataset path
last year
utils
initial commit
last year
.gitignore
initial commit
last year
README.md
docs: update README.md
last year
demo.ipynb
fixed a demo bug
last year
inference_iter.py
fixed a small bug: return an extra None when no <SEG> generated.
11 months ago
merge_lora_weights_and_save_hf_model.py
initial commit
last year
requirements.txt
Update requirements.txt
11 months ago
train_ds.py
add glus_s glus_a and demo
last year
Repository files navigation
README
GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation
Lang Lin*, Xueyang Yu*, Ziqi Pang*, Yu-Xiong Wang

[Project Page] [arXiv]

arXiv Project HuggingFace


Overview
RefVOS in complex scenarios places high demands on models' video understanding and fine-grained localization capabilities. Recently, numerous models leveraging MLLM-based comprehension and reasoning abilities have been proposed to address this challenge. Our GLUS advances further along this methodological path.

🚀 GLUS is principled. It utilizes global-local reasoning to combine holistic video understanding with detailed frames understanding, unleashing the potential of fine-grained segmentation in complex scenarios.

✨ GLUS is powerful. It unifies the methods of memory bank, object contrastive learning and key frame selection to tackle the problems of mask inconsistency and object obfuscation, achieving state-of-the-art performance in complex-scenario RefVOS tasks.

📌 GLUS is simple. It elegantly integrates the approach for complex-scenario RefVOS tasks within a single MLLM framework, eliminating the necessity of utilizing other independent modules.


News
Installation
git clone git@github.com:GLUS-video/GLUS.git && cd GLUS
pip install -r requirements.txt
pip install ./model/segment-anything-2
pip install flash-attn==2.6.2 --no-build-isolation
Model Zoo
For more convenient following, we provide the checkpoints of GLUS without object contrastive learning.

Model	Training Datasets	Methods	Download	MeViS J&F	Ref-Youtube-VOS J&F
GLUSSpartial	MeViS, Ref-Youtube-VOS	GLU + MB	HuggingFace, ModelScope	49.5	65.2
GLUSS	MeViS, Ref-Youtube-VOS	GLU + MB + OC + KFS	HuggingFace, ModelScope	50.3	66.6
GLUSA	+ RefDAVIS17, ReVOS, LVVIS	GLU + MB	HuggingFace, ModelScope	51.3	67.3
Notes: “GLU”: Global-local unification, “MB”: End-to-end memory bank, “OC”: Object contrastive loss, “KFS”: key frame selection. GLUSS refers to the model trained on a subset of existing RefVOS datasets (Mevis and Ref-Youtube-VOS), while GLUSA denotes the model trained on the full set of available datasets.

We recommend to download and store the pretrained weights at GLUS_ROOT/checkpoints.

Training and Validation
1. Data Preparation
Please follow the below architecture to prepare the datasets. We recommend to set DATASET_ROOT to GLUS_ROOT/data.

RefVOS Datasets: MeViS, Refer-YouTube-VOS, Ref-DAVIS17.
Reasoning VOS Datasets: ReVOS, ReasonVOS
Open-Vocabulary Video Instance Segmentation Dataset: LV-VIS.
Datasets Architecture
DATASET_ROOT
├── mevis
│   ├── train
│   │   ├── JPEGImages
│   │   ├── mask_dict.json
│   │   └── meta_expressions.json
│   ├── valid
│   │   ├── JPEGImages
│   │   └── meta_expressions.json
│   └── valid_u
│       ├── JPEGImages
│       ├── mask_dict.json
│       └── meta_expressions.json
├── Refer-YouTube-VOS
│   ├── meta_expressions
│   │   ├── train/meta_expressions.json
│   │   └── valid/meta_expressions.json
│   ├── train
│   │   ├── JPEGImages
│   │   └── Annotations
│   └── valid
│       └── JPEGImages
├── DAVIS17
│   ├── meta_expressions
│   │   ├── train/meta_expressions.json
│   │   └── valid/meta_expressions.json
│   ├── train
│   │   ├── JPEGImages
│   │   └── Annotations
│   └── valid
│       ├── JPEGImages
│       └── Annotations
├── LVVIS
│   ├── train
│   │   └── JPEGImages
│   ├── mask_dict.json
│   └── meta_expressions.json
├── ReVOS
│   ├── JPEGImages 
│   ├── mask_dict.json             
│   ├── mask_dict_foreground.json   
│   ├── meta_expressions_train_.json 
│   └── meta_expressions_valid_.json 
├── ReasonVOS
│   ├── JPEGImages 
│   ├── Annotations           
│   ├── meta_expressions.json 

2. Model Weights Preparation
Follow the guidance to prepare for the pretrained weights of LISA and SAM-2 for training GLUS:

Download the pretrained weights of LISA from LISA-7B-v1.
Download the pretrained weights of SAM-2 from sam2_hiera_large.
Then organize them in the following architecture:
3. Training
Set the paths in the scripts and then run scripts/train_glus_s.sh or scripts/train_glus_a.sh. The scripts will automatically start the training, and transform the saved checkpoint into hugging-face format when the training finished.

Key Frame Selection
For the usage of key frame selection, please refer to the KFS_README.

4. Evaluation
Set the paths, val_set and set_name in scripts/inference.sh, and then run it. It will detect the available GPUs firstly and then individually run parallelizable inference on each gpu.

Evaluation with Key Frame Selection
Set the args use_kf and kf_path in scripts/inference_kf.sh, and then run it. We provide our json file on Mevis and Refyoutube-VOS for GLUSS on the google drive.

After the masks are generated completely, run the corresponding evaluation python file in utils. You may need to set the groundtruth mask path, predicted mask path and expressions json file path. Please refer to the eval files to see the help on arguments.

An example:

python utils/eval_mevis.py \
  --mevis_exp_path='$GLUS_ROOT/data/mevis/valid_u/meta_expressions.json' \
  --mevis_mask_path='$GLUS_ROOT/data/mevis/valid_u/mask_dict.json'
  --mevis_pred_path='$GLUS_ROOT/generated'
Specially, to evaluate the performance on Refer-YouTube-VOS Valid or MeViS Valid benchmarks, you may need to submit the predicted masks results following the guidance at MeViS-Evaluation-Server or RefYoutube-Evaluation-Server.

Inference and Demo
Please refer to demo.ipynb to inference on your own videos and referrings.

For more examples, please refer to our Project Page.

Citation
If you find this work useful in your research, please consider citing:

@inproceedings{lin2025glus,
  title={GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation},
  author={Lin, Lang and Yu, Xueyang and Pang, Ziqi and Wang, Yu-Xiong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
Acknowledgement
We thank the contributors to the following open-source projects. Our project is impossible without the inspirations from these excellent researchers.

LISA
SAM2
Mevis
VISA
About
[CVPR 2025] Official PyTorch Implementation of GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation

glus-video.github.io/
Topics
video-understanding video-segmentation multi-modality referring-expression-segmentation referring-video-object-segmentation multimodal-large-language-models
Resources
 Readme
 Activity
 Custom properties
Stars
 70 stars
Watchers
 1 watching
Forks
 6 forks
Report repository
Releases
No releases published