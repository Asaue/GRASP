import os
import sys
import json
import torch
import numpy as np
import PIL.Image
import types
from tqdm import tqdm
from pycocotools import mask as cocomask

# ================= 配置区域 =================
CONFIG = {
    'data_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    'image_folder': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images',
    'output_dir': '/home/yrquni/Downloads/Unilab/PaDT/eval/outputs/refcrack_rl',
    # SAM2 路径 (请确认此路径存在)
    'sam_checkpoint': '/Data/Docker_liuwu/models/sam2.1-hiera-large/sam2.1_hiera_large.pt',
    'sam_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml'
}
# ===========================================

# 1. 环境与导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from utils import load_model

# 尝试导入 SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    try:
        from sam2.build_sam2 import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except:
        print("[Warning] SAM2 environment not found.")

# ==============================================================================
# [Monkey Patch] 运行时热补丁：修复 padt.py 的崩溃问题
# ==============================================================================

def patched_forward_main(self, input_ids=None, attention_mask=None, **kwargs):
    # 提取参数
    past_image_embeds = kwargs.get('past_image_embeds')
    inputs_embeds = kwargs.get('inputs_embeds')
    image_grid_thw = kwargs.get('image_grid_thw')
    
    # 仅处理 Embedding 构建阶段
    if inputs_embeds is None and kwargs.get('pixel_values') is not None:
        pixel_values = kwargs['pixel_values'].type(self.visual.dtype)
        image_embeds, high_res_image_embeds, visual_pe = self.visual(pixel_values, grid_thw=image_grid_thw)

        if self.use_visual_prototype_projection:
            image_prototypes = self.vis_norm(image_embeds)
            image_prototypes = image_prototypes + self.vis_proj(image_prototypes)
        else:
            image_prototypes = image_embeds.clone()

        embed_tokens = self.model.embed_tokens.weight
        extended_embed_tokens = torch.cat([embed_tokens, image_prototypes], dim=0)

        # 保护 1: ID 越界
        if input_ids.max() >= extended_embed_tokens.shape[0]:
            input_ids = torch.clamp(input_ids, max=extended_embed_tokens.shape[0] - 1)
        
        inputs_embeds = extended_embed_tokens[input_ids]

        # 保护 2: 数量对齐 (防止 CUDA Error)
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        
        if n_image_tokens > 0:
            if n_image_features != n_image_tokens:
                if n_image_features < n_image_tokens: # 补齐
                    diff = n_image_tokens - n_image_features
                    pad = torch.zeros((diff, image_embeds.shape[1]), device=image_embeds.device, dtype=image_embeds.dtype)
                    image_embeds = torch.cat([image_embeds, pad], dim=0)
                else: # 截断
                    image_embeds = image_embeds[:n_image_tokens]
            
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        kwargs['inputs_embeds'] = inputs_embeds
        if 'pixel_values' in kwargs: del kwargs['pixel_values']

    # 调用原始逻辑
    return PaDTForConditionalGeneration._original_forward_main(self, input_ids, attention_mask, **kwargs)

def patched_vl_decode(self, object_vp_feats, low_res_image_embeds, high_res_image_embeds, image_grid_thws, visual_pes):
    cu_object_vp_feat = sum(object_vp_feats, [])
    true_value = len(cu_object_vp_feat) > 0
    
    if true_value:
        # 手动构造输入 (简化版，复用原逻辑太复杂，这里做最小实现)
        # 注意：这里假设 batch 里的图片被展开了
        patch_offset = 0
        cu_low, cu_high, cu_pe0, cu_pe1, cu_patch, obj_thws = [], [], [], [], [], []
        cu_sample_idx = []

        for idx, (obj_feat, thw) in enumerate(zip(object_vp_feats, image_grid_thws)):
            pn = thw.cumprod(dim=-1)[-1].item()
            low = low_res_image_embeds[patch_offset//4 : (patch_offset+pn)//4]
            high = high_res_image_embeds[patch_offset : patch_offset+pn]
            pe0 = visual_pes[0][patch_offset : patch_offset+pn]
            pe1 = visual_pes[1][patch_offset : patch_offset+pn]
            
            n_obj = len(obj_feat)
            cu_sample_idx.extend([idx]*n_obj)
            cu_low.append(low.unsqueeze(0).repeat(n_obj,1,1).flatten(0,1))
            cu_high.append(high.unsqueeze(0).repeat(n_obj,1,1).flatten(0,1))
            cu_pe0.append(pe0.unsqueeze(0).repeat(n_obj,1,1).flatten(0,1))
            cu_pe1.append(pe1.unsqueeze(0).repeat(n_obj,1,1).flatten(0,1))
            cu_patch.extend([pn]*n_obj)
            obj_thws.extend([thw]*n_obj)
            patch_offset += pn

        cu_patch = torch.nn.functional.pad(torch.tensor(cu_patch, device=self.device).cumsum(dim=0), (1,0)).int()
        cu_low = torch.cat(cu_low, dim=0)
        cu_high = torch.cat(cu_high, dim=0)
        pe = (torch.cat(cu_pe0, dim=0), torch.cat(cu_pe1, dim=0))
        obj_thws = torch.stack(obj_thws, dim=0)

        # 调用 Decoder
        ret = self.vl_decoder(cu_object_vp_feat, cu_low, cu_high, pe, cu_patch, obj_thws, self.device)
        
        # 兼容性解包
        if len(ret) == 5:
            bbox, score, mask, hw, pts_local = ret
        else:
            bbox, score, mask, hw = ret
            pts_local = torch.zeros((len(bbox), 2), device=self.device)

        # 计算全局坐标
        PATCH_SIZE = 14 * self.config.vision_config.spatial_merge_size
        _, Ws = hw
        # 这里的 Ws 是每个对象对应的 feature map 宽
        # 重新计算每个对象在 feature map 里的位置
        obj_pn = cu_patch[1:] - cu_patch[:-1]
        total_pn = cu_patch[-1].item()
        
        # 生成全局索引
        offsets = cu_patch[:-1]
        # 这里为了简化，我们直接利用 batch 内的线性索引
        # 实际上上面的 Ws 已经是对齐到每个对象的了
        
        # 既然我们已经把 batch 展开成了 N 个对象，每个对象都有自己的 patch 序列
        # 我们只需要生成 0~pn-1 的序列
        flat_indices = []
        for pn_i in obj_pn:
            flat_indices.append(torch.arange(pn_i, device=self.device))
        flat_indices = torch.cat(flat_indices)
        
        row = flat_indices // Ws.repeat_interleave(obj_pn)
        col = flat_indices % Ws.repeat_interleave(obj_pn)
        
        global_y = (row + pts_local[:, 1]) * PATCH_SIZE
        global_x = (col + pts_local[:, 0]) * PATCH_SIZE
        global_pts = torch.stack([global_x, global_y], dim=-1)
        
        # 分组
        pts_list = []
        start = 0
        for pn_i in obj_pn:
            pts_list.append(global_pts[start : start+pn_i])
            start += pn_i

        return {
            'pred_boxes': bbox, 'pred_score': score, 'pred_mask': mask, 
            'sample_idx': cu_sample_idx, 'pred_points': pts_list
        }
    
    return {'sample_idx': [], 'pred_points': []}

# ==============================================================================

def custom_resize_image(image):
    try:
        w, h = image.size
        if w < 28 or h < 28:
            if w < h: new_w = 28; new_h = int(h * (28 / w))
            else: new_h = 28; new_w = int(w * (28 / h))
            image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    except: pass
    return image

def main():
    # 1. 解析参数
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1] # 从命令行获取路径
    else:
        # 如果没有传入，这里是个默认值，但你运行脚本时一定会传
        checkpoint_path = 'PaDT-MLLM/PaDT_Pro_3B' 

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0: os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 2. 加载模型 (使用传入的 checkpoint_path)
    print(f"[Rank {local_rank}] Loading Model from: {checkpoint_path}")
    model, processor, _ = load_model(checkpoint_path, local_rank)
    processor = VisonTextProcessingClass(processor)
    
    # ⚡️应用热补丁 (覆盖原版逻辑)
    if not hasattr(PaDTForConditionalGeneration, '_original_forward_main'):
        PaDTForConditionalGeneration._original_forward_main = PaDTForConditionalGeneration.forward_main
    
    model.forward_main = types.MethodType(patched_forward_main, model)
    model.vl_decode = types.MethodType(patched_vl_decode, model)
    
    model.eval()

    # 加载 SAM2
    print(f"[Rank {local_rank}] Loading SAM2...")
    try:
        sam2_model = build_sam2(CONFIG['sam_cfg'], CONFIG['sam_checkpoint'], device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
    except:
        sam2_predictor = None
        print("[Warning] SAM2 load failed.")

    # 3. 加载数据
    all_data = []
    with open(CONFIG['data_file'], 'r') as f:
        for line in f:
            if line.strip(): all_data.append(json.loads(line))
    
    my_data = all_data[local_rank::int(os.environ.get("WORLD_SIZE", 1))]
    print(f"[Rank {local_rank}] Processing {len(my_data)} samples")

    output_file = os.path.join(CONFIG['output_dir'], f'rank{local_rank}.json')
    f_out = open(output_file, 'w')

    for item in tqdm(my_data):
        try:
            img_path = os.path.join(CONFIG['image_folder'], item['image'][0] if isinstance(item['image'], list) else item['image'])
            if not os.path.exists(img_path): continue
            
            raw_image = PIL.Image.open(img_path).convert("RGB")
            resized_image = custom_resize_image(raw_image)
            prompt = item['conversations'][0]['value'].replace('<image>', '').strip()

            text = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}], tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[resized_image], padding=True, return_tensors="pt", add_special_tokens=False)
            
            if 'image_grid_thw' in inputs:
                inputs["input_ids"] = processor.assign_to_global_vrt_id(inputs["input_ids"], inputs['image_grid_thw'])
            inputs = inputs.to(device)

            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, do_sample=False, output_hidden_states=True, return_dict_in_generate=True)

            prompt_len = inputs["input_ids"].size(1)
            generated_ids = outputs.sequences[:, prompt_len:]
            _, feats, _, _, _ = parseVRTintoCompletion(processor, generated_ids, outputs.hidden_states, torch.tensor([False], device=device), outputs.past_image_embeds, inputs['image_grid_thw'])
            
            decoded = model.vl_decode(feats, outputs.past_image_embeds, outputs.past_high_res_image_embeds, inputs['image_grid_thw'], outputs.past_visual_pe)
            
            if len(decoded['sample_idx']) == 0: continue

            # SAM2 推理
            if sam2_predictor:
                points = decoded['pred_points'][0].cpu().numpy()
                bbox_norm = decoded['pred_boxes'][0].cpu().tolist()
                
                H, W = inputs['image_grid_thw'][0][1].item()*14, inputs['image_grid_thw'][0][2].item()*14
                cx, cy, w, h = bbox_norm
                box = np.array([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])

                sam2_predictor.set_image(np.array(resized_image))
                masks, _, _ = sam2_predictor.predict(point_coords=points, point_labels=np.ones(len(points)), box=box[None, :], multimask_output=False)
                
                res = {
                    "id": item['id'],
                    "box": box.tolist(),
                    "points": points.tolist(),
                    "mask_rle": cocomask.encode(np.asfortranarray(masks[0].astype(np.uint8)))
                }
                res['mask_rle']['counts'] = res['mask_rle']['counts'].decode('utf-8')
                f_out.write(json.dumps(res) + "\n")
                f_out.flush()

        except Exception as e:
            # traceback.print_exc()
            continue

    f_out.close()
    if torch.distributed.is_initialized(): torch.distributed.barrier()

if __name__ == "__main__":
    main()