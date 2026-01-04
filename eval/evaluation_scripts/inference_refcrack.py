import re
import os
import json
import torch
import torch.distributed as dist
import deepspeed
from datasets import load_dataset, Dataset
from utils import load_model, infer_dataset
from PaDT import VisonTextProcessingClass

# ================= Configuration =================
# 在这里配置你的绝对路径
CONFIG = {
    # 你的 Ground Truth 数据集路径
    'data_file': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/crack_ref_train.json',
    
    # 你的图片文件夹路径
    'image_folder': '/home/yrquni/Downloads/Unilab/PaDT/dataset/dataset_root/images',
    
    # 输出目录
    'output_dir': '../outputs/refcrack'
}
# =================================================

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

local_rank, world_size, rank = setup_distributed()
device = f"cuda:{local_rank}"
print(f"Process {rank} using {device}")

if __name__ == "__main__":
    import sys
    # 接收 Shell 脚本传过来的参数
    if len(sys.argv) > 1:
        checkpoint = sys.argv[1] # 模型路径
        split = sys.argv[2]      # split 名称 (例如 crack_val)
        suffix = sys.argv[3]     # 后缀 (例如 padt_crack)
    else:
        # 默认值 (方便调试)
        checkpoint = 'PaDT-MLLM/PaDT_Pro_3B'
        split = 'crack_val'
        suffix = 'padt_crack'

    model_path = f'{checkpoint}'
    
    # --- 修改点: 使用自定义的 Crack 数据集路径 ---
    data_files = [CONFIG['data_file']] 
    image_folders = [CONFIG['image_folder']]
    output_dir = CONFIG['output_dir']
    
    assert len(data_files) == len(image_folders), "Number of data files must match number of image folders"
    
    # 确保输出目录存在 (仅在主进程创建)
    if rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    print(f"[Rank {rank}] Loading data from {data_files[0]}...")
    
    for data_file, image_folder in zip(data_files, image_folders):
        if os.path.exists(data_file) is False:
            print(f"[Error] Data file not found: {data_file}")
            continue
        
        # 读取 JSONL (Line-by-Line)
        try:
            with open(data_file, 'r') as f:
                data = [json.loads(i) for i in f.readlines()]
        except Exception as e:
            print(f"[Error] Failed to load json: {e}")
            data = []
        
        for item in data:
            if 'image' in item:
                if isinstance(item['image'], str):
                    # 构造绝对图片路径
                    img_path = os.path.join(image_folder, item['image'])
                    item['image_path'] = [img_path]
                    
                    # 检查图片是否存在 (可选，防止报错中断)
                    # if not os.path.exists(img_path):
                    #     continue
                    
                    del item['image'] 
                elif isinstance(item['image'], list):
                    item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                    del item['image'] 
                else:
                    raise ValueError(f"Unsupported image type: {type(item['image'])}")
            
                # 提取 prompt: 从 conversations 里拿
                if 'conversations' in item and len(item['conversations']) > 0:
                    item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                else:
                    item['problem'] = "Please detect the object." # Fallback

                # 清理不需要的字段以节省内存
                item.pop('answer_template', None)
                item.pop('objects', None)
                item.pop('conversations', None)
                
                all_data.append(item)

    print(f"[Rank {rank}] Total samples loaded: {len(all_data)}")
    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        # 这里的 example['image_path'] 已经是列表了
        return {
            'image_path': example['image_path'],
            'problem': example['problem'],
            'solution': None,
            'prompt': [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                    {'type': 'text', 'text': example['problem']}
                ]
            }]
        }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=4) # 稍微降低 num_proc 防止内存溢出
    
    # 加载模型
    print(f"[Rank {rank}] Loading model from {model_path}...")
    model, processor, accelerator = load_model(model_path, local_rank)
    processor = VisonTextProcessingClass(processor)

    # 对齐 vocab size
    with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
        model_embed_token_size = model.model.embed_tokens.weight.shape[0]
    processor.prepare(model_embed_token_size)

    # 开始推理
    print(f"[Rank {rank}] Starting inference...")
    infer_dataset(
        model=model,
        dataset=dataset,
        processor=processor,
        accelerator=accelerator,
        output_dir=output_dir,
        batch_size=32,       # 如果显存不够 (OOM)，请把这里改小，比如 8 或 16
        datasetname=split,   # 这将决定输出文件名的前缀
        suffix=suffix        # 这将决定输出文件名的后缀
    )