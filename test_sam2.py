from PaDT import PaDTForConditionalGeneration, VisonTextProcessingClass, parseVRTintoCompletion
from qwen_vl_utils import process_vision_info
from transformers import Sam2Processor, Sam2Model, AutoProcessor
import os
import torch
import cv2
import numpy as np
from PIL import Image
import re

# -------------------------------
# 路径配置
# -------------------------------
TEST_IMG_PATH = "/home/yrquni/Downloads/Unilab/PaDT/13.png"
MODEL_PATH = "/Data/Docker_liuwu/models/checkpoints/PaDT-REC-3B_crack_1112"
SAM_MODEL_PATH = "/Data/Docker_liuwu/models/facebook/sam2.1-hiera-large"  # 官网推荐路径

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 加载 PaDT 模型
# -------------------------------
model = PaDTForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": 0})
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor = VisonTextProcessingClass(processor, model.config.vision_config.spatial_merge_size)
processor.prepare(model.model.embed_tokens.weight.shape[0])

PROMPT = """Please carefully check the image and detect the object this sentence describes: 
"The crack extending from the middle to the upper right"."""


message = [
    {"role": "user",
     "content": [
         {"type": "image", "image": TEST_IMG_PATH},
         {"type": "text", "text": PROMPT}
     ]}
]

text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(message)

MAX_SIDE = 644
new_image_inputs = []
for image in image_inputs:
    im_w, im_h = image.size
    scale = MAX_SIDE / max(im_w, im_h)
    new_w, new_h = int(im_w * scale), int(im_h * scale)
    new_image_inputs.append(image.resize((new_w, new_h), Image.Resampling.LANCZOS))

prompt_inputs = processor(
    text=[text],
    images=new_image_inputs,
    padding=True,
    padding_side="left",
    return_tensors="pt",
    add_special_tokens=False
).to(device)

# -------------------------------
# PaDT 推理阶段
# -------------------------------
with torch.inference_mode():
    prompt_inputs["input_ids"] = processor.assign_to_global_vrt_id(prompt_inputs["input_ids"], prompt_inputs['image_grid_thw'])
    generate_returned_result = model.generate(
        **prompt_inputs,
        use_cache=True, max_new_tokens=1024, do_sample=False,
        output_hidden_states=True, return_dict_in_generate=True
    )

    prompt_completion_ids = processor.assign_to_local_vrt_id(
        generate_returned_result['sequences'], prompt_inputs['image_grid_thw']
    )

    prompt_length = prompt_inputs["input_ids"].size(1)
    completion_ids = prompt_completion_ids[:, prompt_length:]

    completions, feats, labels, vrts, vrts_feats = parseVRTintoCompletion(
        processor, completion_ids, generate_returned_result['hidden_states'], torch.Tensor([False])
    )

    low_res_image_embeds = generate_returned_result.past_image_embeds
    high_res_image_embeds = generate_returned_result.past_high_res_image_embeds
    visual_pe = generate_returned_result.past_visual_pe

    decoded_list = model.vl_decode(feats, low_res_image_embeds, high_res_image_embeds, prompt_inputs['image_grid_thw'], visual_pe)
    pred_boxes = decoded_list['pred_boxes']
    pred_scores = decoded_list['pred_score'].sigmoid()
    pred_labels = sum(labels, [])
    pred_vrts = sum(vrts, [])
    pred_masks = decoded_list['pred_mask']
    pred_mask_valid_hws = torch.stack([decoded_list['pred_mask_valid_hw'][0], decoded_list['pred_mask_valid_hw'][1]], dim=-1)
    box_2_sample_idx = decoded_list['sample_idx']

# -------------------------------
# VRT patch 中心点提取
# -------------------------------
image = cv2.imread(TEST_IMG_PATH)
im_h, im_w = image.shape[:2]
scale = MAX_SIDE / max(im_w, im_h)
im_w, im_h = int(im_w * scale), int(im_h * scale)
image = cv2.resize(image, (im_w, im_h))

resized_h, resized_w = round(im_h / 28) * 28, round(im_w / 28) * 28
patch_h, patch_w = round(im_h / 28), round(im_w / 28)

all_patch_centers = []

for vrt in pred_vrts:
    vrt_idxs = re.findall(r'<\|VRT_(\d+)\|>', vrt)
    for vrt_idx in vrt_idxs:
        vrt_x, vrt_y = int(vrt_idx) % patch_w, int(vrt_idx) // patch_w
        cx, cy = int((vrt_x + 0.5) * 28), int((vrt_y + 0.5) * 28)
        all_patch_centers.append([cx, cy])

# -------------------------------
# SAM2 点提示分割（官方方式）
# -------------------------------
sam_processor = Sam2Processor.from_pretrained(SAM_MODEL_PATH)
sam_model = Sam2Model.from_pretrained(SAM_MODEL_PATH).to(device)

input_points = [[all_patch_centers]]  # [batch, object, points, 2]
input_labels = [[[1]*len(all_patch_centers)]]  # 全部正点

image_pil = Image.open(TEST_IMG_PATH).convert("RGB").resize((resized_w, resized_h))
sam_inputs = sam_processor(
    images=image_pil,
    input_points=input_points,
    input_labels=input_labels,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    sam_outputs = sam_model(**sam_inputs)

# 后处理 mask
masks = sam_processor.post_process_masks(
    sam_outputs.pred_masks.cpu(),
    sam_inputs["original_sizes"]
)[0].numpy()

# 可视化
sam_mask = np.zeros_like(image)
for m in masks:
    sam_mask[m > 0.5] = [0, 255, 255]

overlay = cv2.addWeighted(image, 0.6, sam_mask, 0.4, 0)
cv2.imwrite("/home/yrquni/Downloads/Unilab/PaDT/outputs/sam_seg.png", overlay)
print("[完成] SAM2 点提示分割已保存到 outputs/sam_seg.png")
