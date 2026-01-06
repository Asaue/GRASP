import os
import sys
import json
import glob
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pycocotools import mask as cocomask
from PIL import Image

# --- 尝试导入 SAM2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    try:
        from sam2.build_sam2 import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("[Error] 无法导入 SAM2，请确保已安装并配置好环境。")
        sys.exit(1)

# ================= 配置区域 =================
# GT 文件路径 (绝对路径)
GT_FILE = '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json'
# 图片文件夹
IMAGE_FOLDER = '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images'
# 预测结果目录 (相对路径)
PRED_DIR = '../outputs/refcrack_rl'
# SAM2 配置
SAM_CHECKPOINT = '/Data/Docker_liuwu/models/sam2.1-hiera-large/sam2.1_hiera_large.pt'
MODEL_CFG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
# ===========================================

def calculate_iou(bbox1, bbox2):
    """计算 Box IoU"""
    x1, y1, w1, h1 = bbox1
    x1_p, y1_p, w1_p, h1_p = bbox2
    b1 = [x1, y1, x1 + w1, y1 + h1]
    b2 = [x1_p, y1_p, x1_p + w1_p, y1_p + h1_p]
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = w1 * h1
    b2_area = w1_p * h1_p
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_mask_ciou(pred_mask, gt_mask):
    """计算 Mask CIoU"""
    p = pred_mask > 0
    g = gt_mask > 0
    intersection = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return intersection / union if union > 0 else 0.0

def main():
    # 1. 接收参数 (兼容 bash 脚本调用)
    if len(sys.argv) < 3:
        print("Usage: python eval_rl_refcrack.py <LOG_SUFFIX> <SPLIT>")
        suffix = 'padt_rl_crack'
        split = 'crack_val_rl'
    else:
        suffix = sys.argv[1]
        split = sys.argv[2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Evaluation Settings ---")
    print(f"Split : {split}")
    print(f"Suffix: {suffix}")
    print(f"Device: {device}")
    print(f"---------------------------")

    # 2. 加载 SAM2 模型
    print(">>> Loading SAM2 Model...")
    try:
        sam2_model = build_sam2(MODEL_CFG, SAM_CHECKPOINT, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 Loaded Successfully.")
    except Exception as e:
        print(f"[Error] SAM2 Loading Failed: {e}")
        return

    # 3. 加载 Ground Truth
    gt_dict = {}
    id_to_filename = {} # 用于根据 ID 找图片文件
    print(f">>> Loading Ground Truth...")
    
    with open(GT_FILE, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                
                # 记录图片文件名
                img_path = item['image']
                if isinstance(img_path, list): img_path = img_path[0]
                id_to_filename[item['id']] = img_path
                
                if 'objects' in item and len(item['objects']) > 0:
                    obj = item['objects'][0]
                    label = obj.get('label', 'crack')
                    bbox_name = '%d_%s' % (item['id'], label)
                    
                    rle = obj['rle']
                    gt_mask = cocomask.decode(rle)
                    h, w = rle['size']
                    norm_box = obj['bbox']
                    abs_box = [norm_box[0]*w, norm_box[1]*h, (norm_box[2]-norm_box[0])*w, (norm_box[3]-norm_box[1])*h]
                    
                    gt_dict[bbox_name] = {'mask': gt_mask, 'bbox': abs_box}
            except: pass
    print(f"Loaded {len(gt_dict)} GT items.")

    # 4. 加载预测结果 (多卡合并)
    # 你的 bash 脚本会产生 split_rank_pred_results_suffix.json
    search_pattern = os.path.join(PRED_DIR, f"{split}_*_pred_results_{suffix}.json")
    pred_files = glob.glob(search_pattern)
    preds = []
    
    print(f">>> Searching for predictions: {search_pattern}")
    for p_file in pred_files:
        with open(p_file, 'r') as f:
            for line in f:
                if line.strip(): preds.append(json.loads(line))

    if not preds:
        print("[Error] No prediction data found! Please check Inference step.")
        sys.exit(1)
    print(f"Loaded {len(preds)} total predictions.")

    # 5. 评估循环 (含 SAM2 Refinement)
    padt_box_ious = []
    padt_mask_cious = []
    sam2_mask_cious = []
    
    print(">>> Start Evaluation & SAM2 Refinement...")
    for i, pred in tqdm(enumerate(preds), total=len(preds)):
        # 匹配 GT
        bbox_name = '%d_%s' % (pred['image_id'], pred['category'])
        if bbox_name not in gt_dict: continue
        gt = gt_dict[bbox_name]
        
        # --- A. PaDT 基准指标 ---
        # Box IoU
        b_iou = calculate_iou(pred['bbox'], gt['bbox'])
        padt_box_ious.append(b_iou)
        
        # Mask CIoU (PaDT Original)
        try:
            padt_mask = cocomask.decode(pred['mask'])
            m_ciou = calculate_mask_ciou(padt_mask, gt['mask'])
            padt_mask_cious.append(m_ciou)
        except:
            padt_mask_cious.append(0.0)
            m_ciou = 0.0

        # --- B. SAM2 精修 ---
        img_filename = id_to_filename.get(pred['image_id'])
        full_img_path = os.path.join(IMAGE_FOLDER, img_filename) if img_filename else None
        
        sam_success = False
        s_ciou = m_ciou # 默认回退到 PaDT 结果

        if full_img_path and os.path.exists(full_img_path):
            try:
                # 1. 读图
                image = Image.open(full_img_path).convert("RGB")
                predictor.set_image(np.array(image))
                
                # 2. 准备 Prompt: Box
                bx, by, bw, bh = pred['bbox']
                input_box = np.array([bx, by, bx+bw, by+bh]) # [x1, y1, x2, y2]
                
                # 3. 准备 Prompt: Points (如果存在)
                input_point = None
                input_label = None
                
                if 'points' in pred and len(pred['points']) > 0:
                    pts = np.array(pred['points'])
                    # 采样防止显存溢出
                    if len(pts) > 20:
                        idx = np.random.choice(len(pts), 20, replace=False)
                        pts = pts[idx]
                    input_point = pts
                    input_label = np.ones(len(pts)) # 1 = 前景
                
                # 4. SAM2 预测
                if input_point is not None:
                    masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        box=input_box[None, :],
                        multimask_output=False
                    )
                else:
                    masks, scores, _ = predictor.predict(
                        box=input_box[None, :],
                        multimask_output=False
                    )
                
                sam_mask = masks[0].astype(np.uint8)
                s_ciou = calculate_mask_ciou(sam_mask, gt['mask'])
                sam_success = True
                
            except Exception as e:
                # print(f"SAM2 Error: {e}")
                pass
        
        sam2_mask_cious.append(s_ciou)

    # 6. 打印最终报表
    if len(padt_box_ious) == 0:
        print("No matched samples found.")
    else:
        padt_ap50 = np.mean(np.array(padt_box_ious) >= 0.5)
        padt_mean_ciou = np.mean(padt_mask_cious)
        sam2_mean_ciou = np.mean(sam2_mask_cious)
        
        print("\n" + "="*50)
        print(f"FINAL EVALUATION REPORT: {split}")
        print("="*50)
        print(f"Samples Evaluated: {len(padt_box_ious)}")
        print("-" * 30)
        print(f"[PaDT Base]")
        print(f"  - REC AP@50 (Box): {padt_ap50*100:.2f}%")
        print(f"  - RES CIoU (Mask): {padt_mean_ciou*100:.2f}%")
        print("-" * 30)
        print(f"[PaDT + SAM2 Refinement]")
        print(f"  - RES CIoU (Mask): {sam2_mean_ciou*100:.2f}%")
        print(f"  - Improvement    : +{(sam2_mean_ciou - padt_mean_ciou)*100:.2f}%")
        print("="*50)

if __name__ == "__main__":
    main()