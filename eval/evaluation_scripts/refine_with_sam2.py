import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as cocomask

# ---【关键修正】修正 Import 路径 ---
# 旧版/部分版本: from sam2.build_sam2 import build_sam2 (你报错的原因)
# 新版/官方主线: from sam2.build_sam import build_sam2 (正确写法)
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    # 备用方案: 如果你的版本非常特殊，尝试旧写法
    from sam2.build_sam2 import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# ================= Configuration =================
CONFIG = {
    # 1. PaDT 预测结果 (你的输入文件)
    'padt_pred_file': '../outputs/refcrack/crack_val_0_pred_results_padt_crack.json',

    # 2. GT 文件
    'gt_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    
    # 3. 图片目录
    'image_folder': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images',
    
    # 4. SAM2 权重路径 (.pt 文件)
    # 必须指向 .pt 文件! 
    # 假设你之前下载在了 sam2-main 目录下，或者 temp_install 目录下
    # 请根据实际情况修改:
    'sam_checkpoint': '/Data/Docker_liuwu/models/sam2.1-hiera-large/sam2.1_hiera_large.pt',
    
    # 5. 模型配置名称
    # 官方库会自动查找 configs/sam2.1/sam2.1_hiera_l.yaml
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',

    # 6. 输出结果文件
    'output_file': '../outputs/refcrack/refined_sam2_results.json'
}
# =================================================

def calculate_ciou(pred: np.ndarray, gt: np.ndarray):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u > 0 else 0.0

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. 加载 SAM2 Native ---
    print(f"Loading SAM2 Native Model...")
    try:
        # build_sam2 第一个参数是 config 路径 (相对 sam2 库根目录)，第二个是 checkpoint
        sam2_model = build_sam2(CONFIG['model_cfg'], CONFIG['sam_checkpoint'], device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 Native Model loaded successfully!")
    except Exception as e:
        print(f"\n[Error] Loading SAM2 failed: {e}")
        print("请检查:")
        print("1. CONFIG['sam_checkpoint'] 是否指向了真实存在的 .pt 文件")
        print("2. CONFIG['model_cfg'] 路径是否正确 (通常是 configs/sam2.1/sam2.1_hiera_l.yaml)")
        return

    # --- 2. 加载 GT 和建立图片索引 ---
    print("Loading Ground Truth...")
    gt_dict = {}
    id_to_filename = {}
    
    with open(CONFIG['gt_file'], 'r') as f:
        # 兼容 JSONL 读取
        lines = f.readlines()
        data = []
        for line in lines:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except: pass
                
        for item in data:
            id_to_filename[item['id']] = item['image']
            
            # 解析 Mask
            if 'objects' in item:
                # Key 必须和 Inference 时生成的一致
                label = item['objects'][0].get('label', 'crack')
                bbox_name = '%d_%s' % (item['id'], label)
                try:
                    rle = item['objects'][0]['rle']
                    # 处理 RLE 格式差异
                    if isinstance(rle, dict):
                        gt_dict[bbox_name] = cocomask.decode(rle)
                    else:
                        # 如果已经是 binary mask 或者其他格式，需自行适配
                        pass 
                except:
                    pass

    # --- 3. 加载 PaDT 预测 ---
    print(f"Loading Predictions from {CONFIG['padt_pred_file']}...")
    preds = []
    with open(CONFIG['padt_pred_file'], 'r') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    # --- 4. 循环精修 ---
    original_cious = []
    refined_cious = []
    new_results = []
    
    print(f"Start Refining {len(preds)} samples...")
    
    for i, pred in tqdm(enumerate(preds), total=len(preds)):
        # Key 匹配
        bbox_name = '%d_%s' % (pred['image_id'], pred['category'])
        if bbox_name not in gt_dict:
            continue
            
        gt_mask = gt_dict[bbox_name]
        
        # 计算旧指标
        try:
            padt_mask = cocomask.decode(pred['mask'])
            old_ciou = calculate_ciou(padt_mask > 0, gt_mask > 0)
            original_cious.append(old_ciou)
        except:
            # 如果解码失败跳过
            continue

        # --- SAM2 核心逻辑 ---
        img_name = id_to_filename.get(pred['image_id'])
        if not img_name: 
            refined_cious.append(old_ciou)
            continue
        
        img_path = os.path.join(CONFIG['image_folder'], img_name)
        if not os.path.exists(img_path):
            refined_cious.append(old_ciou)
            continue
        
        try:
            # 1. 读图并设置给 predictor
            image = Image.open(img_path).convert("RGB")
            predictor.set_image(np.array(image))
            
            # 2. 准备 Prompt (Box)
            # PaDT: [x, y, w, h] -> SAM2: [x1, y1, x2, y2]
            x, y, w, h = pred['bbox']
            input_box = np.array([x, y, x + w, y + h])

            # 3. 预测
            # multimask_output=False 让模型自己选一个最好的 mask
            masks, scores, _ = predictor.predict(
                box=input_box[None, :], 
                multimask_output=False
            )
            
            # masks shape: (1, H, W)
            sam_mask = masks[0].astype(np.uint8)
            
            # 4. 计算新指标
            new_ciou = calculate_ciou(sam_mask > 0, gt_mask > 0)
            refined_cious.append(new_ciou)
            
            # 5. 保存结果
            if CONFIG.get('output_file'):
                rle = cocomask.encode(np.asfortranarray(sam_mask))
                rle['counts'] = rle['counts'].decode('utf-8')
                pred_copy = pred.copy()
                pred_copy['mask'] = rle
                pred_copy['sam2_ciou'] = new_ciou
                new_results.append(pred_copy)
            
        except Exception as e:
            # print(f"Error sample {i}: {e}")
            refined_cious.append(old_ciou)

    # --- 5. 结果打印 ---
    print("\n" + "="*40)
    print(f"SAM2 REFINEMENT RESULTS (Native)")
    print("="*40)
    if len(refined_cious) > 0:
        mean_old = np.mean(original_cious)
        mean_new = np.mean(refined_cious)
        print(f"Samples Evaluated: {len(refined_cious)}")
        print("-" * 20)
        print(f"Original PaDT CIoU   : {mean_old:.4f} ({mean_old*100:.2f}%)")
        print(f"Refined (SAM2) CIoU  : {mean_new:.4f} ({mean_new*100:.2f}%)")
        print(f"Improvement          : +{(mean_new - mean_old)*100:.2f}%")
        
        # 保存文件
        if CONFIG.get('output_file'):
            with open(CONFIG['output_file'], 'w') as f:
                for item in new_results:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved refined results to {CONFIG['output_file']}")
    else:
        print("No samples processed.")
    print("="*40)

if __name__ == "__main__":
    main()