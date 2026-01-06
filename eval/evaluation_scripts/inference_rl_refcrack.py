import os
import json
import torch
import numpy as np
import deepspeed
import math
from tqdm import tqdm
from pycocotools import mask as cocomask
from PaDT import VisonTextProcessingClass, parseVRTintoCompletion
from utils import load_model

# ================= Configuration =================
CONFIG = {
    'data_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    'image_folder': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images',
    'output_dir': '/home/yrquni/Downloads/Unilab/PaDT/eval/outputs/refcrack_rl' # 根据你的Log调整了输出目录
}
# =================================================

def main():
    import sys
    
    # 1. 注入环境变量 (防止单卡运行 utils 报错)
    if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ: os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29500"

    # 2. 参数解析
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
        split = sys.argv[2]
        suffix = sys.argv[3]
    else:
        checkpoint = '/Data/Docker_liuwu/models/checkpoints/PaDT-REC-3B_RL_crack'
        split = 'crack_val_rl'
        suffix = 'padt_crack_points'

    # 3. 初始化设备
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        if not os.path.exists(CONFIG['output_dir']):
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 4. 加载并分发数据 (含 Padding 逻辑)
    print(f"[Rank {local_rank}] Loading data...")
    all_data = []
    try:
        with open(CONFIG['data_file'], 'r') as f:
            for line in f:
                if line.strip():
                    try: all_data.append(json.loads(line))
                    except: pass
    except: pass

    # --- [核心修改] Padding 逻辑 ---
    total_samples = len(all_data)
    # 计算每个 GPU 应处理的最大样本数 (向上取整)
    # 例如 577 / 4 = 144.25 -> max_samples = 145
    samples_per_gpu = math.ceil(total_samples / world_size)

    # 获取当前 Rank 应该处理的索引列表
    # 原始索引: [0, 4, 8...], [1, 5, 9...]
    my_indices = list(range(local_rank, total_samples, world_size))
    
    # 计算需要补多少个 dummy 数据
    num_padding = samples_per_gpu - len(my_indices)
    
    # 构建最终的数据列表 (包含标记)
    # 格式: (item, is_padding)
    my_processing_list = []
    
    # 添加真实数据
    for idx in my_indices:
        my_processing_list.append((all_data[idx], False))
        
    # 添加 Padding 数据 (简单重复最后一个真实数据)
    if num_padding > 0:
        if len(my_indices) > 0:
            last_item = all_data[my_indices[-1]]
            for _ in range(num_padding):
                my_processing_list.append((last_item, True)) # is_padding = True
        else:
            # 极端情况：总数据量少于 GPU 数量，某些 GPU 分不到数据
            # 此时无法复制，只能跳过 (但 DeepSpeed 可能会报错，需确保至少有数据)
            pass

    print(f"[Rank {local_rank}] Real: {len(my_indices)}, Padding: {num_padding}, Total: {len(my_processing_list)}")
    # -----------------------------

    # 5. 加载模型
    model, processor, accelerator = load_model(checkpoint, local_rank)
    processor = VisonTextProcessingClass(processor)
    with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
        model_embed_token_size = model.model.embed_tokens.weight.shape[0]
    processor.prepare(model_embed_token_size)
    model.eval()

    # 6. 推理
    output_filename = f'{split}_{local_rank}_pred_results_{suffix}.json'
    output_path = os.path.join(CONFIG['output_dir'], output_filename)
    f_out = open(output_path, 'w')

    # 使用处理列表进行循环
    for item, is_padding in tqdm(my_processing_list, disable=(local_rank != 0)):
        # 即便是 Padding 数据，也要跑完整个模型前向过程，以维持多卡同步
        try:
            image_id = item['id']
            img_name = item['image'][0] if isinstance(item['image'], list) else item['image']
            image_path = os.path.join(CONFIG['image_folder'], img_name)
            
            if not os.path.exists(image_path): continue

            human_input = item['conversations'][0]['value'] if 'conversations' in item else "Please detect."
            prompt_text = human_input.replace('<image>', '').strip()

            message = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(message)
            
            inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", add_special_tokens=False).to(device)

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)
                
                # 如果是 Padding 数据，跑完 generate 就可以停了，不需要解析和写入
                if is_padding:
                    continue

                prompt_len = inputs["input_ids"].size(1)
                generated_ids = outputs.sequences[:, prompt_len:]
                
                completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(
                    processor, generated_ids, outputs.hidden_states, 
                    torch.tensor([False], device=device), outputs.past_image_embeds, inputs['image_grid_thw']
                )

                decoded = model.vl_decode(
                    feats, outputs.past_image_embeds, outputs.past_high_res_image_embeds, 
                    inputs['image_grid_thw'], outputs.past_visual_pe
                )

                if len(decoded['sample_idx']) > 0:
                    img_h = inputs['image_grid_thw'][0][1].item()
                    img_w = inputs['image_grid_thw'][0][2].item()
                    
                    pred_box_norm = decoded['pred_boxes'][0].tolist()
                    pred_score = float(decoded['pred_score'][0].sigmoid().item())
                    
                    cx, cy, bw, bh = pred_box_norm
                    x = (cx - bw/2) * img_w
                    y = (cy - bh/2) * img_h
                    w = bw * img_w
                    h = bh * img_h
                    abs_bbox = [x, y, w, h]

                    mask_h_valid = decoded['pred_mask_valid_hw'][0][0].item()
                    mask_w_valid = decoded['pred_mask_valid_hw'][1][0].item()
                    raw_mask = decoded['pred_mask'][0, :mask_h_valid*4, :mask_w_valid*4]
                    mask_np = (raw_mask > 0).float().cpu().numpy().astype(np.uint8)
                    rle = cocomask.encode(np.asfortranarray(mask_np))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    points_list = decoded['pred_points'][0].cpu().tolist() if 'pred_points' in decoded and len(decoded['pred_points']) > 0 else []

                    result_item = {
                        "image_id": image_id,
                        "category": prompt_text,
                        "bbox": abs_bbox,
                        "score": pred_score,
                        "mask": rle,
                        "points": points_list 
                    }
                    f_out.write(json.dumps(result_item) + "\n")
                    f_out.flush()

        except Exception as e:
            # print(f"Error: {e}")
            continue

    f_out.close()
    
    # 7. 最后加一个 Barrier，确保所有 Rank 都跑完了再退出
    if torch.distributed.is_initialized():
        print(f"[Rank {local_rank}] Waiting for other ranks...")
        torch.distributed.barrier()
    
    print(f"[Rank {local_rank}] Finished. Saved to {output_path}")

if __name__ == "__main__":
    main()