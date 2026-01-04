import os
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as cocomask

# ================= Configuration =================
# 在这里配置你的绝对路径，避免命令行传参错误
CONFIG = {
    # 你的 Ground Truth 数据集路径
    'gt_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    
    # 预测结果输出目录
    'output_dir': '../outputs/refcrack',
    
    # 预测结果的标识符 (对应 shell 脚本中的 LOG_SUFFIX)
    'suffix': 'padt_pro_3b',
    
    # 当前跑的 Split 名称 (对应 shell 脚本中的 SPLIT)
    # 注意：如果你的预测结果文件名是 crack_val_0_pred_....json，这里就填 crack_val
    'split': 'crack_val' 
}
# =================================================

def calculate_iou(bbox1, bbox2):
    """
    bbox1: GT [x, y, w, h] (Absolute)
    bbox2: Pred [x, y, w, h] (Absolute)
    """
    x1, y1, w1, h1 = bbox1
    x1_prime, y1_prime, w1_prime, h1_prime = bbox2

    # Convert to [x1, y1, x2, y2] for intersection calc
    bbox1_coords = [x1, y1, x1 + w1, y1 + h1]
    bbox2_coords = [x1_prime, y1_prime, x1_prime + w1_prime, y1_prime + h1_prime]

    inter_x1 = max(bbox1_coords[0], bbox2_coords[0])
    inter_y1 = max(bbox1_coords[1], bbox2_coords[1])
    inter_x2 = min(bbox1_coords[2], bbox2_coords[2])
    inter_y2 = min(bbox1_coords[3], bbox2_coords[3])

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    bbox1_area = w1 * h1
    bbox2_area = w1_prime * h1_prime
    union_area = bbox1_area + bbox2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def calculate_ciou(pred: np.ndarray, gt: np.ndarray):
    i = np.logical_and(pred, gt).sum()
    u = np.logical_or(pred, gt).sum()
    return i/u if u > 0 else 0.0

if __name__ == "__main__":
    import sys
    
    # 优先使用命令行参数，如果没有则使用上方 CONFIG
    suffix = sys.argv[1] if len(sys.argv) > 1 else CONFIG['suffix']
    split = sys.argv[2] if len(sys.argv) > 2 else CONFIG['split']
    
    output_dir = CONFIG['output_dir']
    gt_file_path = CONFIG['gt_file']

    print(f"--- Evaluation Settings ---")
    print(f"Split : {split}")
    print(f"Suffix: {suffix}")
    print(f"GT    : {gt_file_path}")
    print(f"---------------------------")

    # 1. 加载预测结果 (Predictions)
    # 假设是分布式的8个文件，如果不确定，代码会自动跳过不存在的
    log_result_paths = [os.path.join(output_dir, f'{split}_{i}_pred_results_{suffix}.json') for i in range(8)]
    
    preds = []
    print(f"Loading predictions from: {output_dir}")
    for log_result_file in log_result_paths:
        if not os.path.exists(log_result_file):
            continue
        try:
            with open(log_result_file, 'r') as f:
                for line in f:
                    if line.strip():
                        preds.append(json.loads(line))
        except Exception as e:
            print(f"Error reading {log_result_file}: {e}")

    if not preds:
        print("Error: No prediction files found or files are empty!")
        sys.exit(1)
    print(f"Loaded {len(preds)} predictions.")

    # 2. 加载 Ground Truth (JSONL Line-by-Line)
    gt_dict = defaultdict(list)
    accuracy = defaultdict(int)
    mask_cious = defaultdict(float)

    print(f"Loading Ground Truth...")
    if not os.path.exists(gt_file_path):
        print(f"Error: GT file not found at {gt_file_path}")
        sys.exit(1)

    with open(gt_file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
                
                # ------ 核心修改: 解析你的数据格式 ------
                # ID
                img_id = item['id']
                
                # 对象信息
                obj = item['objects'][0]
                label = obj['label'] # Prompt text
                
                # 构造唯一 Key: 必须和 Inference 输出的格式一致
                # 通常是 ID_Text
                bbox_name = '%d_%s' % (img_id, label)

                # 获取尺寸: 直接从 RLE 中拿 [H, W]，不需要读图！
                # item['objects'][0]['rle']['size'] 通常是 [H, W]
                # 注意 pycocotools 习惯 (H, W)
                img_h, img_w = obj['rle']['size'] 

                # 解析 Mask
                rle = obj['rle']
                gt_mask = cocomask.decode(rle)

                # 解析 BBox 并反归一化
                # 你的数据: [0.024, 0.545, 0.704, 0.685] -> [x1, y1, x2, y2] Normalized
                bbox_norm = obj['bbox']
                
                # 转换为绝对坐标 [x, y, w, h]
                x1 = bbox_norm[0] * img_w
                y1 = bbox_norm[1] * img_h
                x2 = bbox_norm[2] * img_w
                y2 = bbox_norm[3] * img_h
                
                w = x2 - x1
                h = y2 - y1
                gt_bbox = [x1, y1, w, h]

                gt_dict[bbox_name] = [gt_bbox, gt_mask]
                accuracy[bbox_name] = 0. # Init score

            except Exception as e:
                # print(f"Skipping line {idx}: {e}")
                pass

    print(f"Loaded {len(gt_dict)} GT items.")

    # 3. 计算指标
    matched_count = 0
    for pred in preds:
        # 构造 Key 以匹配 GT
        # 确保 inference 代码中生成的 category 字段就是 prompt 文本
        bbox_name = '%d_%s' % (pred['image_id'], pred['category'])
        
        if bbox_name in gt_dict:
            matched_count += 1
            gt_bbox, gt_mask = gt_dict[bbox_name]
            
            pred_bbox = pred['bbox']
            # pred mask 也是 RLE 编码的，需要 decode
            pred_mask = cocomask.decode(pred['mask'])

            # CIoU (Mask IoU)
            ciou = calculate_ciou(pred_mask > 0, gt_mask > 0)
            
            # Box IoU
            iou = calculate_iou(gt_bbox, pred_bbox)

            # 更新最高分
            accuracy[bbox_name] = max(iou, accuracy[bbox_name])
            mask_cious[bbox_name] = max(ciou, mask_cious[bbox_name])

    # 4. 打印结果
    all_ious = np.array([i for i in accuracy.values()])
    all_mask_cious = np.array([i for i in mask_cious.values()])

    print('\n' + '='*30)
    if matched_count == 0:
        print("WARNING: No predictions matched with Ground Truth!")
        print("Please check if 'bbox_name' format matches between GT and Preds.")
        print(f"Sample GT Key  : {list(gt_dict.keys())[0] if gt_dict else 'None'}")
        print(f"Sample Pred Key: {'%d_%s' % (preds[0]['image_id'], preds[0]['category']) if preds else 'None'}")
    else:
        ap = (all_ious >= 0.5).mean()
        mean_cious = all_mask_cious.mean()
        
        print(f'Matched Samples : {matched_count}')
        print(f'Total GT Samples: {len(gt_dict)}')
        print('-'*30)
        print(f'REC AP_50 : {ap:.4f} ({ap*100:.2f}%)')
        print(f'RES CIoU  : {mean_cious:.4f} ({mean_cious*100:.2f}%)')
    print('='*30 + '\n')