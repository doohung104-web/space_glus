import argparse
import os
import sys
import json
from PIL import Image
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.GLUS import GLUSForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from sam2.utils.transforms import SAM2Transforms
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, CONTEXT_INFO_LIST, SHORT_QUESTION_LIST)

from dataset.mevis import load_mevis_json
from dataset.refyoutube_vos import load_refyoutube_json
from dataset.revos import load_revos_json
from dataset.davis17 import load_davis17_json
from dataset.reasonvos import load_reason_json

from utils.trajectory_utils import (
    DEFAULT_HISTORY_LEN,
    STATE_DIM,
    mask_to_box_state,
)


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLUS eval")
    parser.add_argument("--version", default="Swindl/GLUS-A") 
    parser.add_argument("--dataset_dir", default='./data', type=str)
    parser.add_argument("--use_kf", action="store_true", default=False)
    parser.add_argument("--score_json", default=None, type=str)
    parser.add_argument("--vis_save_path", default="./generated", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--context_frame_num", default=4, type=int)
    parser.add_argument("--question_frame_num", default=4, type=int)
    parser.add_argument("--set_name", default="valid_u", type=str)
    parser.add_argument("--model_arch", default="4seg_proj", type=str)
    parser.add_argument("--val_set", default="mevis", type=str)
    parser.add_argument("--image_features_num", default=63, type=int)
    parser.add_argument("--mem_stride", default=3, type=int)
    parser.add_argument("--not_use_mem_bank", action="store_true", default=False)
    parser.add_argument("--sam_config", default="sam2_hiera_l.yaml", type=str)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # ----- Phase-2A trajectory inference MVP -----
    parser.add_argument(
        "--use_trajectory",
        action="store_true",
        default=False,
        help="Enable Phase-2A trajectory conditioning on the forward inference "
             "path.  Requires a checkpoint trained with the trajectory branch.",
    )
    parser.add_argument(
        "--traj_history_len",
        type=int,
        default=DEFAULT_HISTORY_LEN,
        help="Length T_h of the per-expression sliding history window built "
             "from previous predicted masks (forward path only).",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )
    kwargs.update({
        "not_use_mem_bank": args.not_use_mem_bank,
    })

    # Phase-2A: ``GLUSForCausalLM.__init__`` reads ``use_trajectory`` from the
    # kwargs (not from the config) and overwrites ``config.use_trajectory``
    # with that value.  Without an explicit kwarg, a merged HF checkpoint
    # that has ``trajectory_encoder.*`` weights would silently drop them
    # ("Some weights ... were not used").  We forward the CLI flag only -
    # other ``traj_*`` knobs fall back to the defaults baked into the model
    # ``__init__`` (which match the values used during training).  When the
    # CLI flag is False we deliberately pass nothing, so the construction
    # path is bit-identical to the original GLUS.
    if getattr(args, "use_trajectory", False):
        kwargs.update({
            "use_trajectory": True,
            "traj_history_len": int(args.traj_history_len),
        })

    model = GLUSForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=False, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx,
        sam_config=args.sam_config, image_features_num=args.image_features_num, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = SAM2Transforms(args.image_size, mask_threshold=0.0)

    model.eval()
    
    flag = True
    while flag is not None:
        flag = inference_video(args, model, clip_image_processor, transform, tokenizer)
    
def get_frame_paths_and_target_obj(args):
    
    if args.val_set == "mevis":
        vid_list, metas, _ = load_mevis_json(args.dataset_dir, set_name=args.set_name)
    elif args.val_set == "refyoutube_vos":
        vid_list, metas, _ = load_refyoutube_json(args.dataset_dir, set_name=args.set_name)
    elif args.val_set == "revos":
        vid_list, metas, _ = load_revos_json(args.dataset_dir, set_name=args.set_name)
    elif args.val_set == 'davis17':
        vid_list, metas, _ = load_davis17_json(args.dataset_dir, set_name=args.set_name)
    elif args.val_set == 'reasonvos':
        vid_list, metas, _ = load_reason_json(args.dataset_dir, set_name=args.set_name)
    else:
        raise ValueError()
    vid2refs = {}
    for ref in metas:
        video_id = ref["video"]
        vid2refs[video_id] = vid2refs.get(video_id, []) + [
            ref,
        ]
            
    for vid_name in vid_list:
        for ref in vid2refs[vid_name]:
            dataset_settings = {
                "vid_name": vid_name,
                "exp_id": ref['exp_id'],
            }
            cur_path = f'{args.vis_save_path}/{dataset_settings["vid_name"]}/{dataset_settings["exp_id"]}'
            
            if os.path.exists(cur_path):
                continue
            else:
                print('Currently generate: ', cur_path)
                os.makedirs(cur_path, exist_ok=True)
                return ref['file_names'], ref['exp'], dataset_settings, ref['frames'] # per target_obj now
    return None, None, None, None
    
    
def inference_video(args, model, clip_image_processor, transform, tokenizer):
    
    image_paths, target_obj, dataset_settings, mask_name = get_frame_paths_and_target_obj(args)
    if image_paths is None:
        return None
    target_obj = target_obj[:-1] if target_obj[-1] == '.' else target_obj
    
    max_index = 0
    if args.use_kf and args.score_json is not None:
        with open(args.score_json, 'r') as f:
            score_json = json.load(f)
        cur_key = dataset_settings['vid_name'] + '_' + dataset_settings['exp_id']
        if cur_key not in score_json:
            max_index = 0
        else:
            cur_fname = int(score_json[cur_key])
            all_fnames = [int(x.split(".")[0]) for x in mask_name]
            all_fnames.sort()
            max_index = all_fnames.index(cur_fname)
        
    context_frame_num = args.context_frame_num
    question_frame_num = args.question_frame_num
    
    frame_num = len(image_paths)
    frame_clips = []
    indices = np.linspace(0, frame_num, context_frame_num + 1, dtype=int)
    
    for i in range(math.ceil(frame_num / context_frame_num)):
        curr_frame_clips = []
        for j in range(context_frame_num):
            curr_frame_clips.append(min(indices[j] + i, indices[j + 1] - 1))
        frame_clips.append(curr_frame_clips)
    
    #forward inference process   
    for idx in range(len(frame_clips) // 2, len(frame_clips) // 2 + 1):
        
        total_frame_list = frame_clips[idx] + list(range(frame_num))[max_index:]
        
        text_output, mask_scores, masks_list, _ = inference_frames(args, model, 
                                                               clip_image_processor = clip_image_processor, 
                                                               transform = transform,
                                                               tokenizer = tokenizer,
                                                               image_paths = [image_paths[i] for i in total_frame_list], 
                                                               target_obj = target_obj)
        
        assert len(masks_list) == len(mask_name) - max_index
        for i in range(len(masks_list)):
            
            assert masks_list[i].shape[0] == 1
            
            for j in range(masks_list[i].shape[0]):
                        
                mask_tensor = masks_list[i][j]
                mask_np = mask_tensor.cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8) 
                mask_img = Image.fromarray(mask_np)
                
                index = max_index + i            
                saved_path = f'{args.vis_save_path}/{dataset_settings["vid_name"]}/{dataset_settings["exp_id"]}/{mask_name[i].split(".")[0]}.png'
                dir_path = os.path.dirname(saved_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                mask_img.save(saved_path)
    
    if max_index == 0:
        return True
              
    #backward inference process
    for idx in range(len(frame_clips) // 2, len(frame_clips) // 2 + 1):
        

        back_index = list(range(frame_num))[:max_index + 1]
        back_index = back_index[::-1]
        total_frame_list = frame_clips[idx] + back_index
        
        # Phase-2A: the backward path is left structurally untouched and
        # therefore always runs without trajectory conditioning, regardless
        # of ``--use_trajectory``.  The trajectory encoder was trained on
        # forward-time histories only and we do not want to feed it a
        # reversed-time sequence in this MVP.
        text_output, mask_scores, masks_list, _ = inference_frames(args, model, 
                                                               clip_image_processor = clip_image_processor, 
                                                               transform = transform,
                                                               tokenizer = tokenizer,
                                                               image_paths = [image_paths[i] for i in total_frame_list], 
                                                               target_obj = target_obj,
                                                               enable_traj=False)
        
        assert len(masks_list) == max_index + 1
        for i in range(len(masks_list)):
            
            assert masks_list[i].shape[0] == 1
            
            for j in range(masks_list[i].shape[0]):
                
                mask_tensor = masks_list[i][j]
                mask_np = mask_tensor.cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8) 
                mask_img = Image.fromarray(mask_np)

                #backforward index
                index = max_index - i
                saved_path = f'{args.vis_save_path}/{dataset_settings["vid_name"]}/{dataset_settings["exp_id"]}/{int(mask_name[index].split(".")[0]):05d}.png'
                dir_path = os.path.dirname(saved_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                mask_img.save(saved_path)
                        
    return True
    
        
def _extract_bin_mask_2d(t):
    """Return a CPU 2-D bool tensor from a predicted-mask tensor.

    Robust to the two shapes that show up on the forward inference path:
    ``[1, H, W]`` (the normal SAM2 output, single expression / single frame)
    and ``[H, W]`` (defensive fallback).  Any other rank is collapsed by
    indexing the leading axis with 0 until at most 2 dims remain; in the
    current single-expression inference path this branch is unreachable
    because ``pred_masks`` is asserted to be 1x1.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    while t.dim() > 2:
        t = t[0]
    if t.dim() < 2:
        # Should not happen, but keep state extraction safe by returning an
        # empty mask rather than crashing.
        return torch.zeros((1, 1), dtype=torch.bool)
    return (t > 0).cpu()


def _update_traj_history(bin_mask_2d, traj_history, prev_valid_state, T_h, state_dim):
    """Append a 10-dim state derived from ``bin_mask_2d`` to ``traj_history``.

    Velocity components (``vx, vy, darea``) follow the same convention as
    ``utils.trajectory_utils.masks_to_state_sequence``: they are differences
    against the previous frame whose ``q`` was 1 (i.e. non-empty mask).  When
    no such reference exists yet, velocities are 0.  Returns the updated
    ``prev_valid_state`` (unchanged when the current mask is empty).
    """
    s7 = mask_to_box_state(bin_mask_2d)
    cx = float(s7[0]); cy = float(s7[1]); w = float(s7[2])
    h = float(s7[3]); area = float(s7[4]); ratio = float(s7[5])
    q = float(s7[6])
    if q > 0.0 and prev_valid_state is not None:
        vx = max(-1.0, min(1.0, cx - float(prev_valid_state[0])))
        vy = max(-1.0, min(1.0, cy - float(prev_valid_state[1])))
        darea = max(-1.0, min(1.0, area - float(prev_valid_state[4])))
    else:
        vx = 0.0
        vy = 0.0
        darea = 0.0
    state10 = torch.tensor(
        [cx, cy, w, h, area, ratio, vx, vy, darea, q],
        dtype=torch.float32,
    )
    if state10.shape[0] != state_dim:
        # The Phase-1 utility always emits STATE_DIM=10.  If the checkpoint
        # config ever changes the state dim we silently keep the window
        # consistent by truncating / zero-padding rather than crashing.
        if state10.shape[0] > state_dim:
            state10 = state10[:state_dim]
        else:
            pad = torch.zeros(state_dim - state10.shape[0], dtype=torch.float32)
            state10 = torch.cat([state10, pad], dim=0)
    traj_history.append(state10)
    if len(traj_history) > T_h:
        traj_history.pop(0)
    if q > 0.0:
        return state10
    return prev_valid_state


def inference_frames(args, model, clip_image_processor, transform, tokenizer, image_paths, target_obj, enable_traj=True):
    
    if args.not_use_mem_bank is False:
        model.clear_mem_bank()
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    frame_num = len(image_paths)
    assert frame_num > args.context_frame_num
    vid_len = frame_num - args.context_frame_num
    
    context_frame_num = args.context_frame_num
    question_frame_num = args.question_frame_num
    
    i = 0
    masks_list = []
    mask_confidence_score_list = []
    
    # pre-dealing video frames
    
    full_images, full_images_clip = [], []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            exit(0)
            
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform(image_np).contiguous()
        resize_list = [image.shape[:2]]

        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()
        
        full_images.append(image)
        full_images_clip.append(image_clip)

    # ----- Phase-2A trajectory sliding window (forward path only) -----
    # ``traj_enabled`` flips on only when the CLI flag, the model flag, and
    # an actually-constructed trajectory encoder all agree.  Anything else
    # keeps the original GLUS inference path untouched.
    traj_enabled = (
        bool(enable_traj)
        and bool(getattr(args, "use_trajectory", False))
        and bool(getattr(model, "use_trajectory", False))
        and getattr(model.model, "trajectory_encoder", None) is not None
    )
    T_h = int(getattr(args, "traj_history_len", DEFAULT_HISTORY_LEN))
    D_s = int(getattr(model, "traj_state_dim", STATE_DIM))
    traj_history = []            # list of CPU float32 [D_s] tensors; rightmost = newest
    prev_valid_state = None      # last 10-dim state with q>0 (None until first non-empty mask)

    while i < vid_len:
        
        i += 1
    
        curr_q_num = (context_frame_num + i) - max(context_frame_num, context_frame_num + i - question_frame_num)
    
        images = full_images[:context_frame_num] + full_images[
            max(context_frame_num, context_frame_num + i - question_frame_num):
                (context_frame_num + i)
        ]
        
        images_clip = full_images_clip[:context_frame_num] + full_images_clip[
            max(context_frame_num, context_frame_num + i - question_frame_num):
                (context_frame_num + i)
        ]
        
        images = torch.stack(images, dim=0).unsqueeze(0).cuda() # add bs = 1
        images_clip = torch.stack(images_clip, dim=0).unsqueeze(0).cuda()
        
        answer_prompt = "Sure, the segmentation result is "
        
        if i == 1:
            q_prompt = CONTEXT_INFO_LIST[0].format(class_name=target_obj.lower())
            q_prompt += SHORT_QUESTION_LIST[0].format(class_name=target_obj.lower())
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                q_prompt = q_prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], q_prompt)
            conv.append_message(conv.roles[1], answer_prompt)
            
        else:
            q_prompt = SHORT_QUESTION_LIST[0].format(class_name=target_obj.lower())
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                q_prompt = q_prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            conv.append_message(conv.roles[0], q_prompt)
            conv.append_message(conv.roles[1], answer_prompt)
            
            if i > question_frame_num:
                conv.messages = conv.messages[2:]
                q_prompt = CONTEXT_INFO_LIST[0].format(class_name=target_obj.lower())
                q_prompt += SHORT_QUESTION_LIST[0].format(class_name=target_obj.lower())
                if args.use_mm_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
                    q_prompt = q_prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                conv.messages[0][-1] = q_prompt
                        
        prompt = "</s>".join(conv.get_prompt().split("</s>")[:-1])
        
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # Build the [1, T_h, D_s] / [1, T_h] trajectory inputs from the
        # sliding window.  Empty history -> pass ``None`` so ``evaluate()``
        # short-circuits cleanly (no broadcasting, no padding-only tensors).
        traj_states_in = None
        traj_valid_mask_in = None
        if traj_enabled and len(traj_history) > 0:
            n_hist = len(traj_history)
            n_keep = min(T_h, n_hist)
            recent = traj_history[-n_keep:]
            pad = T_h - n_keep
            if pad > 0:
                pad_states = [torch.zeros(D_s, dtype=torch.float32) for _ in range(pad)]
                states_full = pad_states + recent
            else:
                states_full = recent
            states_t = torch.stack(states_full, dim=0).unsqueeze(0)
            valid_t = torch.cat(
                [
                    torch.zeros(pad, dtype=torch.float32),
                    torch.ones(n_keep, dtype=torch.float32),
                ],
                dim=0,
            ).unsqueeze(0)
            # dtype/device are realigned to ``text_embeds`` inside evaluate();
            # we just put them on GPU here to avoid an extra .cuda() in the
            # critical path.
            traj_states_in = states_t.cuda()
            traj_valid_mask_in = valid_t.cuda()

        output_ids, pred_masks, outputs = model.evaluate(
            images_clip,
            images,
            input_ids,
            resize_list,
            original_size_list,
            rel_pos_list=list(range(max(0, i - question_frame_num), i)),
            mask_clips_list=None,
            max_new_tokens=1,
            tokenizer=tokenizer,
            context_frame_num=context_frame_num,
            question_frame_num=curr_q_num,
            mem_stride=args.mem_stride,
            decode_iter=True,
            traj_states=traj_states_in,
            traj_valid_mask=traj_valid_mask_in,
        )
        
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        
        last_text = tokenizer.decode(output_ids[-1:], skip_special_tokens=False).strip()

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False).strip()
        text_output = text_output.replace("\n", "").replace("  ", " ")
        
        if pred_masks == None and outputs == None:
            print(f"Failed to generate image {i - 1} in current video. Auto-fill an [SEG] and blank mask here.")
            conv.messages[-1][-1] += '[SEG] .'
            mask = torch.zeros(original_size_list[-1])
            masks_list.append(mask.unsqueeze(0))
            if traj_enabled:
                # Keep the sliding window in lock-step with masks_list even
                # on the LLM-generation failure path: write a q=0 state so
                # the next iteration still sees a meaningful temporal slot.
                bin_mask_2d = _extract_bin_mask_2d(mask)
                prev_valid_state = _update_traj_history(
                    bin_mask_2d, traj_history, prev_valid_state, T_h, D_s
                )
            continue
        
        assert len(pred_masks) == 1
        assert len(pred_masks[0]) == 1
            
        masks_list.append((pred_masks[-1][0] > 0).int())

        if traj_enabled:
            bin_mask_2d = _extract_bin_mask_2d(masks_list[-1])
            prev_valid_state = _update_traj_history(
                bin_mask_2d, traj_history, prev_valid_state, T_h, D_s
            )

        conv.messages[-1][-1] = text_output.split("ASSISTANT: ")[-1].split("</s>")[0] + '.'
        
    return text_output, pred_masks, masks_list, mask_confidence_score_list
    
if __name__ == "__main__":
    main(sys.argv[1:])

