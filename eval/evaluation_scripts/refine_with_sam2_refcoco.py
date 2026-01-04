import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from pycocotools import mask as cocomask

# --- Import SAM2 ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    from sam2.build_sam2 import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Import Datasets (关键：引入 huggingface datasets) ---
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    sys.exit(1)

# ================= Configuration =================
CONFIG = {
    'suffix': 'padt_pro_3b',
    'split': 'refcoco_val',
    'output_dir': '../outputs/refcoco',
    
    # 这里的路径哪怕不存在也没关系，脚本会自动去 HuggingFace 下载
    'gt_path_hint': 'PaDT-MLLM/RefCOCO', 
    
    'image_root': '../../dataset/coco/train2014',
    'sam_checkpoint': '/Data/Docker_liuwu/models/sam2.1-hiera-large/sam2.1_hiera_large.pt', 
    'model_cfg': 'configs/sam2.1/sam2.1_hiera_l.yaml',
    'use_point_prompt': True,
    'num_points': 6
}
# =================================================

def calculate_ciou(pred_mask, gt_mask):
    i = np.logical_and(pred_mask, gt_mask).sum()
    u = np.logical_or(pred_mask, gt_mask).sum()
    return i/u if u > 0 else 0.0

def sample_points_from_mask(mask, num_points=5):
    y_idxs, x_idxs = np.where(mask > 0)
    if len(y_idxs) == 0: return None, None
    if len(y_idxs) <= num_points:
        indices = np.arange(len(y_idxs))
    else:
        indices = np.random.choice(len(y_idxs), num_points, replace=False)
    points = np.column_stack([x_idxs[indices], y_idxs[indices]])
    labels = np.ones(len(points))
    return points, labels

if __name__ == "__main__":
    if len(sys.argv) > 1:
        suffix = sys.argv[1]
        split = sys.argv[2]
    else:
        suffix = CONFIG['suffix']
        split = CONFIG['split']

    output_dir = CONFIG['output_dir']
    image_root = CONFIG['image_root']
    
    # 构建预测结果路径
    pred_file_path = os.path.join(output_dir, f'{split}_0_pred_results_{suffix}.json')

    print(f"--- SAM2 Refine Settings ---")
    print(f"Split : {split}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"----------------------------")

    # 1. 加载 SAM2
    print(f"Loading SAM2 model...")
    try:
        sam2_model = build_sam2(CONFIG['model_cfg'], CONFIG['sam_checkpoint'], device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        sys.exit(1)

    # 2. 加载 Ground Truth (复刻 eval_refcoco.py 的逻辑)
    print("Loading Ground Truth...")
    gt_dict = {}
    id_to_filename = {}
    
    # === 关键修改：优先尝试本地，失败则通过 HuggingFace 加载 ===
    local_gt_path = os.path.join(CONFIG['gt_path_hint'], f"{split}.json")
    
    if os.path.exists(local_gt_path):
        print(f"Loading from local file: {local_gt_path}")
        with open(local_gt_path, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]
    else:
        print(f"Local file not found. Loading from HuggingFace Hub: {CONFIG['gt_path_hint']}...")
        try:
            # 这行代码完全照搬原本的 eval_refcoco.py
            # 它会自动下载并缓存，包含完整的 masks 信息
            dataset = load_dataset(CONFIG['gt_path_hint'], data_files=f"{split}.json")
            data = dataset['train'].to_list()
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            sys.exit(1)
    # ========================================================

    # 解析 GT 数据
    for item in data:
        # ID 解析
        item_id = item.get('id', item.get('image_id', item.get('ref_id')))
        if item_id is None: continue

        # 图片路径映射
        img_name = item.get('image', item.get('file_name'))
        if img_name:
            id_to_filename[item_id] = os.path.basename(img_name)

        # Label 解析
        label = None
        if 'objects' in item and len(item['objects']) > 0:
            label = item['objects'][0].get('label')
        elif 'sentences' in item:
            label = item['sentences'][0].get('raw')
        else:
            label = item.get('normal_caption')

        # Mask 解析 (HuggingFace 数据集里通常有 objects/rle)
        if label:
            bbox_name = '%d_%s' % (item_id, label)
            try:
                if 'objects' in item and 'rle' in item['objects'][0]:
                    rle = item['objects'][0]['rle']
                    gt_dict[bbox_name] = cocomask.decode(rle)
            except: pass

    print(f"Loaded {len(gt_dict)} GT masks (with valid RLE).")
    
    if len(gt_dict) == 0:
        print("Error: Still loaded 0 masks. The dataset source might be incorrect.")
        sys.exit(1)

    # 3. 加载预测结果
    print(f"Loading predictions from {pred_file_path}...")
    preds = []
    if os.path.exists(pred_file_path):
        with open(pred_file_path, 'r') as f:
            for line in f:
                if line.strip(): preds.append(json.loads(line))
    else:
        print(f"Error: Pred file not found.")
        sys.exit(1)

    # 4. Refine 过程
    original_cious = []
    refined_cious = []
    
    print(f"Start Refining {len(preds)} samples...")
    for i, pred in tqdm(enumerate(preds), total=len(preds)):
        bbox_name = '%d_%s' % (pred['image_id'], pred['category'])
        
        if bbox_name not in gt_dict: continue
            
        gt_mask = gt_dict[bbox_name]
        
        # 原始精度
        try:
            padt_mask = cocomask.decode(pred['mask'])
            old_ciou = calculate_ciou(padt_mask > 0, gt_mask > 0)
            original_cious.append(old_ciou)
        except: continue
        
        # SAM2 Refine
        img_filename = id_to_filename.get(pred['image_id'])
        if not img_filename: 
            refined_cious.append(old_ciou); continue
            
        img_path = os.path.join(image_root, img_filename)
        # 路径容错
        if not os.path.exists(img_path):
            img_path = os.path.join(image_root, os.path.basename(img_filename))
        if not os.path.exists(img_path):
            refined_cious.append(old_ciou); continue

        try:
            image = Image.open(img_path).convert("RGB")
            predictor.set_image(np.array(image))
            
            x, y, w, h = pred['bbox']
            box = np.array([x, y, x+w, y+h])
            
            points, labels = None, None
            if CONFIG['use_point_prompt']:
                points, labels = sample_points_from_mask(padt_mask, CONFIG['num_points'])
            
            if points is not None:
                masks, scores, _ = predictor.predict(point_coords=points, point_labels=labels, box=box[None, :], multimask_output=True)
            else:
                masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
            
            best_overlap = -1
            best_idx = 0
            for m_idx in range(3):
                overlap = calculate_ciou(masks[m_idx].astype(np.uint8), padt_mask)
                if overlap > best_overlap:
                    best_overlap = overlap; best_idx = m_idx
            
            sam_mask = masks[best_idx].astype(np.uint8)
            new_ciou = calculate_ciou(sam_mask > 0, gt_mask > 0)
            refined_cious.append(new_ciou)
            
        except Exception:
            refined_cious.append(old_ciou)

    # 5. 结果
    print("\n" + "="*40)
    if len(refined_cious) > 0:
        mean_old = np.mean(original_cious)
        mean_new = np.mean(refined_cious)
        print(f"PaDT CIoU      : {mean_old:.4f}")
        print(f"PaDT+SAM2 CIoU : {mean_new:.4f}")
        print(f"Improvement    : {(mean_new - mean_old)*100:+.2f}%")
    else:
        print("No matches found.")
    print("="*40 + "\n")