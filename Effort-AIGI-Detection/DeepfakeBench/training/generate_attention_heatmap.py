"""
生成effort模型的注意力热力图脚本

Usage:
    python generate_attention_heatmap.py \
        --weights /path/to/effort_weights.pth \
        --output_dir ./attention_heatmaps
"""

import os
import numpy as np
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as pil_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
import argparse
from typing import Tuple, List, Dict

from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== Attention Hook ==============
class AttentionExtractor:
    """用于提取ViT各层注意力权重的钩子"""
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子到所有自注意力层"""
        self.attention_maps = []
        self.hooks = []
        
        # CLIP ViT的encoder结构
        for layer_idx, layer in enumerate(self.model.backbone.encoder.layers):
            hook = layer.self_attn.register_forward_hook(self._get_attention_hook(layer_idx))
            self.hooks.append(hook)
    
    def _get_attention_hook(self, layer_idx):
        """创建attention hook"""
        def hook(module, input, output):
            # CLIP的self_attn输出是(hidden_states, attention_weights, ...)
            # 但是默认情况下attention weights不会返回，需要修改forward
            # 我们需要手动计算attention
            pass
        return hook
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_attention_from_clip_vit(model, image_tensor):
    """
    从CLIP ViT模型提取注意力权重
    使用output_attentions=True参数
    """
    model.eval()
    with torch.no_grad():
        # CLIP的vision model支持output_attentions参数
        outputs = model.backbone(
            image_tensor,
            output_attentions=True,
            return_dict=True
        )
        # attentions: tuple of (batch, num_heads, seq_len, seq_len) for each layer
        attentions = outputs.attentions
        
    return attentions


def attention_rollout(attentions, discard_ratio=0.0, head_fusion='mean'):
    """
    Attention Rollout: 将多层注意力矩阵相乘以获取token到CLS token的整体注意力
    
    Args:
        attentions: list of attention tensors, shape (batch, num_heads, seq_len, seq_len)
        discard_ratio: 丢弃最低注意力比例
        head_fusion: 'mean', 'max', 'min' - 如何融合多头
    
    Returns:
        attention map: (batch, seq_len)
    """
    result = None
    
    for attention in attentions:
        # attention: (batch, num_heads, seq_len, seq_len)
        if head_fusion == 'mean':
            attention_heads_fused = attention.mean(dim=1)  # (batch, seq_len, seq_len)
        elif head_fusion == 'max':
            attention_heads_fused = attention.max(dim=1)[0]
        elif head_fusion == 'min':
            attention_heads_fused = attention.min(dim=1)[0]
        else:
            raise ValueError(f"Unknown head fusion: {head_fusion}")
        
        # 丢弃低注意力值
        if discard_ratio > 0:
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * (1 - discard_ratio)), -1, largest=True)
            indices = indices.sort(dim=-1)[0]
            
            for batch_idx in range(attention_heads_fused.size(0)):
                flat[batch_idx] = flat[batch_idx].index_select(-1, indices[batch_idx])
        
        # 添加残差连接 (identity)
        I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
        I = I.unsqueeze(0).expand(attention_heads_fused.size(0), -1, -1)
        
        a = (attention_heads_fused + I) / 2  # 添加残差连接
        a = a / a.sum(dim=-1, keepdim=True)  # 重新归一化
        
        if result is None:
            result = a
        else:
            result = torch.bmm(a, result)
    
    # 获取CLS token到所有patch的注意力
    # result: (batch, seq_len, seq_len)
    # 第一个token是CLS token
    mask = result[:, 0, 1:]  # 排除CLS token本身
    
    return mask


def get_attention_map_last_layer(attentions, head_fusion='mean'):
    """
    使用最后一层的注意力作为热力图
    """
    # 取最后一层
    attention = attentions[-1]  # (batch, num_heads, seq_len, seq_len)
    
    if head_fusion == 'mean':
        attention_fused = attention.mean(dim=1)  # (batch, seq_len, seq_len)
    elif head_fusion == 'max':
        attention_fused = attention.max(dim=1)[0]
    else:
        attention_fused = attention.mean(dim=1)
    
    # CLS token对所有patch的注意力
    cls_attention = attention_fused[:, 0, 1:]  # (batch, num_patches)
    
    return cls_attention


def visualize_attention(attention_map, image, save_path, alpha=0.5):
    """
    将注意力热力图叠加到原图上并保存
    
    Args:
        attention_map: (num_patches,) or (H_patches, W_patches)
        image: 原始图像 (H, W, 3) BGR格式
        save_path: 保存路径
        alpha: 热力图透明度
    """
    # ViT-L/14 对于224x224图像: patch_size=14, 16x16=256 patches
    num_patches = attention_map.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    # reshape attention map to 2D
    attention_map_2d = attention_map.reshape(grid_size, grid_size)
    
    # 上采样到原图大小
    h, w = image.shape[:2]
    attention_map_resized = cv2.resize(attention_map_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 归一化到0-255
    attention_map_normalized = (attention_map_resized - attention_map_resized.min()) / \
                                (attention_map_resized.max() - attention_map_resized.min() + 1e-8)
    attention_map_uint8 = (attention_map_normalized * 255).astype(np.uint8)
    
    # 应用colormap
    heatmap = cv2.applyColorMap(attention_map_uint8, cv2.COLORMAP_JET)
    
    # BGR转RGB用于显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 叠加
    overlay = (image_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    
    # 创建图像: 原图 | 热力图 | 叠加图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    im = axes[1].imshow(attention_map_normalized, cmap='jet')
    axes[1].set_title('Attention Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  [✓] Saved: {save_path}")


def load_model(detector_cfg: str, weights: str):
    """加载effort模型"""
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print("[✓] Effort model loaded.")
    return model


def preprocess_image(image_path: str, target_size: int = 224):
    """加载并预处理图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # resize
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 转换为tensor
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    tensor = transform(pil_image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×H×W
    
    return img_resized, tensor


def generate_heatmap_for_image(
    model,
    image_path: str,
    output_dir: str,
    method: str = 'rollout',  # 'rollout' or 'last_layer'
    head_fusion: str = 'mean'
):
    """
    为单张图像生成注意力热力图
    """
    # 加载图像
    img, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)
    
    # 获取注意力权重
    attentions = get_attention_from_clip_vit(model, tensor)
    
    # 计算注意力热力图
    if method == 'rollout':
        attention_map = attention_rollout(attentions, discard_ratio=0.0, head_fusion=head_fusion)
    else:  # last_layer
        attention_map = get_attention_map_last_layer(attentions, head_fusion=head_fusion)
    
    # 转换为numpy
    attention_map = attention_map[0].cpu().numpy()  # (num_patches,)
    
    # 生成保存路径
    img_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{img_name}_attention_{method}.png")
    
    # 可视化并保存
    visualize_attention(attention_map, img, save_path)
    
    return attention_map


def main():
    parser = argparse.ArgumentParser(description="Generate attention heatmaps for effort model")
    parser.add_argument("--detector_config", type=str, 
                        default="config/detector/effort.yaml",
                        help="Path to detector config yaml")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to effort model weights")
    parser.add_argument("--output_dir", type=str, default="./attention_heatmaps",
                        help="Output directory for heatmaps")
    parser.add_argument("--method", type=str, default="rollout",
                        choices=['rollout', 'last_layer'],
                        help="Method to compute attention: 'rollout' or 'last_layer'")
    parser.add_argument("--head_fusion", type=str, default="mean",
                        choices=['mean', 'max', 'min'],
                        help="How to fuse multi-head attention")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args.detector_config, args.weights)
    
    # 你指定的6张图片路径
    image_paths = [
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5460_71.png", "Asian-Male_F"),
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7184_271.png", "White-Male_F"),
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6477_221.png", "Black-Female_F_0_6477"),
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6710_211.png", "Asian-Female_R"),
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5447_81.png", "White-Female_F"),
        ("/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7430_201.png", "Black-Female_F_0_7430"),
    ]
    
    print(f"\n{'='*60}")
    print(f"Generating attention heatmaps for {len(image_paths)} images")
    print(f"Method: {args.method}, Head fusion: {args.head_fusion}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    for img_path, label in image_paths:
        print(f"Processing: {label}")
        print(f"  Path: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"  [!] Warning: Image not found, skipping...")
            continue
        
        # 加载图像
        img, tensor = preprocess_image(img_path)
        tensor = tensor.to(device)
        
        # 获取注意力权重
        attentions = get_attention_from_clip_vit(model, tensor)
        
        # === 生成多种热力图 ===
        
        # 1. Attention Rollout
        attention_rollout_map = attention_rollout(attentions, discard_ratio=0.0, head_fusion=args.head_fusion)
        attention_rollout_map = attention_rollout_map[0].cpu().numpy()
        save_path_rollout = os.path.join(args.output_dir, f"{label}_attention_rollout.png")
        visualize_attention(attention_rollout_map, img, save_path_rollout)
        
        # 2. Last layer attention
        attention_last = get_attention_map_last_layer(attentions, head_fusion=args.head_fusion)
        attention_last = attention_last[0].cpu().numpy()
        save_path_last = os.path.join(args.output_dir, f"{label}_attention_last_layer.png")
        visualize_attention(attention_last, img, save_path_last)
        
        # 3. 中间层注意力 (第12层, 共24层)
        mid_layer_idx = len(attentions) // 2
        attention_mid = attentions[mid_layer_idx]
        if args.head_fusion == 'mean':
            attention_mid = attention_mid.mean(dim=1)
        else:
            attention_mid = attention_mid.max(dim=1)[0]
        attention_mid = attention_mid[:, 0, 1:].cpu().numpy()[0]
        save_path_mid = os.path.join(args.output_dir, f"{label}_attention_mid_layer.png")
        visualize_attention(attention_mid, img, save_path_mid)
        
        print()
    
    print(f"\n{'='*60}")
    print(f"All heatmaps saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

