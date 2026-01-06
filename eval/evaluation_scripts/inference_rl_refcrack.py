import os
import sys
import json
import torch
import PIL.Image
import numpy as np
import traceback  # 引入 traceback 用于打印详细堆栈
from tqdm import tqdm
from pycocotools import mask as cocomask

# 添加 src 目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from PaDT import VisonTextProcessingClass, parseVRTintoCompletion
from utils import load_model

# ================= Configuration =================
CONFIG = {
    'data_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    'image_folder': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images',
    'output_dir': '/home/yrquni/Downloads/Unilab/PaDT/eval/outputs/refcrack_rl'
}
# =================================================

def custom_resize_image(image):
    """
    复用训练代码中的 Resizing 逻辑
    """
    try:
        w, h = image.size
        if w < 28 or h < 28:
            if w < h:
                new_w = 28
                new_h = int(h * (28 / w))
            else:
                new_h = 28
                new_w = int(w * (28 / h))
            image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    except:
        pass
    return image

def main():
    # 1. 环境变量
    if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = "1"
    if "RANK" not in os.environ: os.environ["RANK"] = "0"
    if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = "29500"

    if len(sys.argv) > 1:
        checkpoint = sys.argv[1]
        split = sys.argv[2]
        suffix = sys.argv[3]
    else:
        checkpoint = 'PaDT-MLLM/PaDT_Pro_3B'
        split = 'crack_val'
        suffix = 'padt_crack_points'

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # 2. 加载数据
    print(f"[Rank {local_rank}] Loading data...")
    all_data = []
    try:
        with open(CONFIG['data_file'], 'r') as f:
            for line in f:
                if line.strip(): all_data.append(json.loads(line))
    except Exception as e:
        print(f"[Error] Load data failed: {e}")
        return

    my_data = all_data[local_rank::world_size]
    print(f"[Rank {local_rank}] Processing {len(my_data)} samples.")

    # 3. 加载模型
    try:
        model, processor, _ = load_model(checkpoint, local_rank)
        processor = VisonTextProcessingClass(processor)
        model.eval()
    except Exception as e:
        print(f"[Error] Model load failed: {e}")
        traceback.print_exc()
        return

    # 4. 推理循环
    output_filename = f'{split}_{local_rank}_pred_results_{suffix}.json'
    output_path = os.path.join(CONFIG['output_dir'], output_filename)
    f_out = open(output_path, 'w')
    
    success_count = 0
    error_count = 0

    iterator = tqdm(my_data) if local_rank == 0 else my_data

    print(f"[Rank {local_rank}] Start Inference Loop...")

    for i, item in enumerate(iterator):
        try:
            image_id = item.get('id', 'unknown')
            img_name = item['image'][0] if isinstance(item['image'], list) else item['image']
            image_path = os.path.join(CONFIG['image_folder'], img_name)
            
            if not os.path.exists(image_path):
                print(f"[Warning] Image not found: {image_path}")
                continue

            # --- 图片处理 ---
            raw_image = PIL.Image.open(image_path).convert("RGB")
            resized_image = custom_resize_image(raw_image)

            human_input = item['conversations'][0]['value'] if 'conversations' in item else "Please detect."
            prompt_text = human_input.replace('<image>', '').strip()

            # --- 构造输入 ---
            text_input = processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=[text_input],
                images=[resized_image], 
                padding=True,
                return_tensors="pt",
                add_special_tokens=False
            )
            
            # --- ID 对齐 ---
            if 'image_grid_thw' in inputs:
                inputs["input_ids"] = processor.assign_to_global_vrt_id(
                    inputs["input_ids"], 
                    inputs['image_grid_thw']
                )
            
            inputs = inputs.to(device)

            # --- 生成 ---
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    use_cache=True,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True
                )

                prompt_len = inputs["input_ids"].size(1)
                generated_ids = outputs.sequences[:, prompt_len:]
                
                # --- 解析 ---
                completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(
                    processor, generated_ids, outputs.hidden_states, 
                    torch.tensor([False], device=device), 
                    outputs.past_image_embeds, 
                    inputs['image_grid_thw']
                )

                decoded = model.vl_decode(
                    feats, outputs.past_image_embeds, outputs.past_high_res_image_embeds, 
                    inputs['image_grid_thw'], outputs.past_visual_pe
                )

                # --- 结果提取 ---
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

                    points_list = []
                    # 检查 'pred_points' 是否存在
                    if 'pred_points' in decoded:
                        if len(decoded['pred_points']) > 0:
                            points_list = decoded['pred_points'][0].cpu().tolist()
                    else:
                        # 如果没有 pred_points，手动抛出异常以便调试
                        raise KeyError("Decoder output does not contain 'pred_points'. Check padt.py modification!")

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
                    success_count += 1
                else:
                    # 打印一条警告，说明没有检测到物体
                    # print(f"[Info] No objects detected for ID {image_id}")
                    pass

        except Exception as e:
            error_count += 1
            # 仅打印前 3 个错误的详细堆栈，防止刷屏
            if error_count <= 3:
                print(f"\n[ERROR] Failed at index {i}, Image ID: {image_id}")
                print(f"Error Message: {str(e)}")
                traceback.print_exc()
            continue

    f_out.close()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    print(f"[Rank {local_rank}] Done. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()