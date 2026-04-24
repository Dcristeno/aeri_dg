import os
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
import re  # 用于清理非法文件名字符
from datasets.aeripedes import AERIPEDES
import matplotlib.image as mpimg

config_file  = 'logs/AERI-PEDES/20251112_234157_finetune/configs.yaml'
args = load_train_configs(config_file)
args.batch_size = 1
args.training = False
device = "cuda:0"
test_img_loader, test_txt_loader, _ = build_dataloader(args)
model = build_model(args)
checkpointer = Checkpointer(model)
checkpointer.load(f=op.join(args.output_dir, 'best0.pth'))
model.to(device)

evaluator = Evaluator(test_img_loader, test_txt_loader)

qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())
qfeats = F.normalize(qfeats, p=2, dim=1)  # text features
gfeats = F.normalize(gfeats, p=2, dim=1)  # image features

similarity = qfeats @ gfeats.t()
_, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)  # q * topk

dataset = AERIPEDES(root='/data1/Datasets/ReID/')
test_dataset = dataset.test

img_paths = test_dataset['img_paths']
captions = test_dataset['captions']
gt_img_paths = test_dataset['img_paths']

def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, rgb_img_paths, gt_rgb_img_paths):
    query_caption = captions[idx]
    query_id = qids[idx]
    rgb_image_paths = [rgb_img_paths[j] for j in indices[idx]]
    image_ids = gids[indices[idx]]
    gt_rgb_image_path = gt_rgb_img_paths[idx]
    return query_id, image_ids, query_caption, rgb_image_paths, gt_rgb_image_path

def plot_retrieval_images(query_id, image_ids, query_caption, rgb_image_paths, gt_rgb_img_path, fname=None):
    print(query_id)
    print(image_ids)
    print(query_caption)
    
    fig = plt.figure(figsize=(12, 6))  # Adjust figure size for better display
    col = len(rgb_image_paths)
    
    # 只展示检索结果中的 RGB 图像
    for i in range(col):
        plt.subplot(2, col, i+1)
        img = Image.open(rgb_image_paths[i])
        ax = plt.gca()  # Get the current axis
        bwith = 4  # Border width
        ax.spines['top'].set_linewidth(bwith)  # Set the border width
        ax.spines['right'].set_linewidth(bwith)
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        
        # 标记与查询图像的匹配情况
        if image_ids[i] == query_id:
            ax.spines['top'].set_color('lawngreen')
            ax.spines['right'].set_color('lawngreen')
            ax.spines['bottom'].set_color('lawngreen')
            ax.spines['left'].set_color('lawngreen')
        else:
            ax.spines['top'].set_color('red')
            ax.spines['right'].set_color('red')
            ax.spines['bottom'].set_color('red')
            ax.spines['left'].set_color('red')
        
        img = img.resize((128, 256))
        plt.imshow(img)
        plt.xticks([])  # Hide x-axis ticks
        plt.yticks([])  # Hide y-axis ticks
        plt.title(f"RGB {i+1}")  # Optional: Title for clarity
    
    fig.tight_layout()  # Adjust layout to avoid overlap
    fig.show()
    
    if fname:
        plt.savefig(fname, dpi=300)

# 清理caption并保存检索结果
output_dir = 'retrieval_results_aeri_baseline'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 使用正则替换非法字符
def clean_caption(caption):
    # 只保留字母、数字和空格，其他字符都替换为下划线
    return re.sub(r'[\\/*?:"<>|]', "_", caption)

# 生成简短文件名
def generate_short_filename(caption, max_length=50):
    # 清理caption并截取前50个字符
    cleaned_caption = clean_caption(caption)
    return cleaned_caption[:max_length]

# 遍历数据集中的所有caption
for idx in range(len(captions)):
    query_id, image_ids, query_caption, rgb_image_paths, gt_rgb_image_path = get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, gt_img_paths)
    
    # 生成简短文件名
    short_filename = generate_short_filename(query_caption)
    
    # 生成保存路径
    fname = os.path.join(output_dir, f"{short_filename}_result.png")
    
    # 调用绘制函数并保存图像
    plot_retrieval_images(query_id, image_ids, query_caption, rgb_image_paths, gt_rgb_image_path, fname=fname)
