import os
import re
import json
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------------------
data_root = "/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/"
trans_model_path = r"/Data/Docker_liuwu/models/Qwen2.5-7B-Instruct"
images_dir = os.path.join(data_root, "images")
masks_dir = os.path.join(data_root, "masks")
ann_path = os.path.join(data_root, "crack_seg_multi100_dan_30_sub_mask_info_eval_3.json")
output_json = os.path.join(data_root, "crack_ref_train.json")

# PaDT 默认 Patch 大小
PATCH_SIZE = 28 

os.makedirs(os.path.dirname(output_json), exist_ok=True)

# -----------------------------------------------------------------------------
# 模型加载
# -----------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(trans_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    trans_model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
model.eval()

# -----------------------------------------------------------------------------
# 数据读取与断点续传
# -----------------------------------------------------------------------------
with open(ann_path, "r") as f:
    data = json.load(f)

finished_ids = set()
if os.path.exists(output_json):
    with open(output_json, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                finished_ids.add(obj["id"])
            except json.JSONDecodeError:
                continue

if finished_ids:
    print(f"🔁 检测到 {len(finished_ids)} 条已完成数据，将自动跳过。")

f_out = open(output_json, "a", encoding="utf-8")

# -----------------------------------------------------------------------------
# 主循环
# -----------------------------------------------------------------------------
for image_id, entry in enumerate(tqdm(data, desc="Processing entries", ncols=100)):
    if image_id in finished_ids:
        continue

    try:
        # 1. 图像读取与 Resize 计算
        image_path = os.path.join(images_dir, entry["image_id"] + ".png")
        image = Image.open(image_path)
        image_w, image_h = image.size
        # 确保尺寸是 PATCH_SIZE 的倍数，符合 PaDT 处理逻辑
        resized_h, resized_w = int(round(image_h / PATCH_SIZE) * PATCH_SIZE), int(round(image_w / PATCH_SIZE) * PATCH_SIZE)

        if entry["type"] == "negative":
            continue

        # 2. Mask 读取与预处理
        mask_path = os.path.join(masks_dir, entry["mask_id"] + ".png")
        m = np.array(Image.open(mask_path).convert("L"))
        m = (m > 128).astype(np.uint8)

        if m.sum() < 10:
            continue

        # 3. 计算 BBox (基于原始 Mask)
        ys, xs = np.where(m > 0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bbox = [x1, y1, x2 - x1, y2 - y1]

        # 4. Resize Mask 并计算 Patch Mask
        # 注意：这里 resized_m 用于后续计算重心，保持灰度值(0或255)以减少精度损失，或者二值化也可
        resized_m = cv2.resize(m * 255, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算哪些 Patch 是有效的 (Intersected)
        # 逻辑：切分成网格 -> 计算每个网格的平均值 -> 阈值判断
        patch_mask = resized_m.reshape(resized_h // PATCH_SIZE, PATCH_SIZE, resized_w // PATCH_SIZE, PATCH_SIZE) \
            .transpose(0, 2, 1, 3).mean(axis=-1).mean(axis=-1) > (255 * 0.05) # 阈值可微调，原代码是 255/28 约等于 9
        
        patch_indices = np.where(patch_mask.reshape(-1))[0].tolist()

        # ---------------------------------------------------------------------
        # [新增] 5. 计算 Patch Points (重心)
        # ---------------------------------------------------------------------
        patch_points = []
        grid_w = resized_w // PATCH_SIZE
        
        for pid in patch_indices:
            # 计算该 Patch 在 Grid 中的行列
            row = pid // grid_w
            col = pid % grid_w
            
            # 截取该 Patch 对应的 Resize 后 Mask 区域
            py1, px1 = row * PATCH_SIZE, col * PATCH_SIZE
            py2, px2 = py1 + PATCH_SIZE, px1 + PATCH_SIZE
            
            local_mask_patch = resized_m[py1:py2, px1:px2]
            
            # 计算重心 (Centroid)
            M = cv2.moments(local_mask_patch)
            if M["m00"] != 0:
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]
            else:
                # 兜底：如果被选为 patch 但 mask 像素极少导致 m00 为 0，取中心
                cX, cY = PATCH_SIZE / 2.0, PATCH_SIZE / 2.0
            
            # 归一化到 [0, 1]
            norm_x = cX / PATCH_SIZE
            norm_y = cY / PATCH_SIZE
            
            # 截断以防万一
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            
            patch_points.append([norm_x, norm_y])
        # ---------------------------------------------------------------------

        # 6. 生成 RLE
        save_rle = mask_utils.encode(np.asfortranarray(m))
        save_rle['counts'] = save_rle['counts'].decode()

        # 7. 模型翻译 (Prompt 生成)
        prompt_text = (
            "你是一个专业的翻译器，只负责中英文互译，不输出任何解释或附加内容。\n"
            "要求：只输出译文本身，不要题目、不要多余话。\n\n"
            f"中文：{entry['prompt']}\n英文："
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"英文[:：]\s*([A-Za-z0-9 ,.'\"-]+)", text)
        if match:
            text = match.group(1).strip()
        else:
            # 清理可能的 prompt 回显，根据 Qwen 的输出来看，有时需要 split
            if "英文：" in text:
                text = text.split("英文：")[-1].strip()
            text = text.strip()

        tqdm.write(f"[{image_id}] {entry['prompt']} → {text}")

        # 8. 构造输出 Item
        item = {
            "id": image_id,
            "image": entry["image_id"] + ".png",
            "conversations": [
                {
                    "from": "human",
                    "value": f"Please carefully check the image and detect the object this sentence describes: \"{text}\"."
                }
            ],
            "task": "refering",
            "answer_template": f"The \"{text}\" refers to <|Obj_0|> in this image.",
            "objects": [
                {
                    "patches": patch_indices,
                    "patch_points": patch_points,  # [新增] 存入数据
                    "bbox": [
                        bbox[0] / image_w,
                        bbox[1] / image_h,
                        (bbox[0] + bbox[2]) / image_w,
                        (bbox[1] + bbox[3]) / image_h
                    ],
                    "iscrowd": 0,
                    "area": int(m.sum()),
                    "rle": save_rle,
                    "label": text
                }
            ]
        }

        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        f_out.flush()

    except Exception as e:
        tqdm.write(f"[ERROR] image_id {image_id}: {e}")
        continue

f_out.close()
print(f"\n✅ 已生成或追加完成，输出文件：{output_json}")