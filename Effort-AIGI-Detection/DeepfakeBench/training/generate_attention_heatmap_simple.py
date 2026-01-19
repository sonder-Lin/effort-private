"""
生成effort模型最后一层注意力热力图

方法: 提取最后一层的attention weights，取CLS token对所有patch的注意力，对所有heads平均

Usage:
    cd /path/to/Effort-AIGI-Detection/DeepfakeBench/training
    python generate_attention_heatmap_simple.py --weights /path/to/your/effort_weights.pth
"""

import os
import sys
import numpy as np
import cv2
import yaml
import torch
from PIL import Image as pil_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import DETECTOR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 你的6张图片 ====================
IMAGE_LIST = [
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5460_71.png",
        "label": "Asian-Male_Fake"
    },
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7184_271.png",
        "label": "White-Male_Fake"
    },
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6477_221.png",
        "label": "Black-Female_Fake_6477"
    },
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6710_211.png",
        "label": "Asian-Female_Real"
    },
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5447_81.png",
        "label": "White-Female_Fake"
    },
    {
        "path": "/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7430_201.png",
        "label": "Black-Female_Fake_7430"
    },
]


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
    print(f"[✓] Effort模型加载成功 (device: {device})")
    return model


def preprocess_image(image_path: str, target_size: int = 224):
    """加载并预处理图像"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    tensor = transform(pil_image.fromarray(img_rgb)).unsqueeze(0)
    
    return img_resized, tensor


def get_last_layer_attention(model, image_tensor):
    """
    获取最后一层的attention weights
    - 取CLS token对所有patch的注意力
    - 对所有heads平均
    
    Returns:
        attention_map: (num_patches,) numpy array
    """
    model.eval()
    with torch.no_grad():
        outputs = model.backbone(
            image_tensor,
            output_attentions=True,
            return_dict=True
        )
        # attentions: tuple of (batch, num_heads, seq_len, seq_len) for each layer
        attentions = outputs.attentions
        
        # 取最后一层
        last_layer_attn = attentions[-1]  # (batch, num_heads, seq_len, seq_len)
        
        # 对所有heads平均
        attn_avg_heads = last_layer_attn.mean(dim=1)  # (batch, seq_len, seq_len)
        
        # 取CLS token (index 0) 对所有patch的注意力，排除CLS自身
        cls_attn = attn_avg_heads[:, 0, 1:]  # (batch, num_patches)
        
    return cls_attn[0].cpu().numpy()  # (num_patches,)


def visualize_and_save(attention_map, image, save_path, title="", alpha=0.5):
    """可视化注意力热力图并保存"""
    # ViT-L/14: 224x224图像, patch_size=14, 得到16x16=256个patches
    num_patches = attention_map.shape[0]
    grid_size = int(np.sqrt(num_patches))
    
    attention_map_2d = attention_map.reshape(grid_size, grid_size)
    
    h, w = image.shape[:2]
    attention_resized = cv2.resize(attention_map_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 归一化到0-1
    att_min, att_max = attention_resized.min(), attention_resized.max()
    attention_norm = (attention_resized - att_min) / (att_max - att_min + 1e-8)
    attention_uint8 = (attention_norm * 255).astype(np.uint8)
    
    # 生成热力图
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = (image_rgb * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    
    # 绘制3张图: 原图 | 热力图 | 叠加图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original', fontsize=14)
    axes[0].axis('off')
    
    im = axes[1].imshow(attention_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Attention Heatmap (Last Layer)', fontsize=14)
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="生成effort模型最后一层注意力热力图")
    parser.add_argument("--detector_config", type=str, 
                        default="config/detector/effort.yaml",
                        help="检测器配置文件路径")
    parser.add_argument("--weights", type=str, required=True,
                        help="effort模型权重路径")
    parser.add_argument("--output_dir", type=str, default="./attention_heatmaps",
                        help="输出目录")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    print("\n" + "="*60)
    print("加载effort模型...")
    print("="*60)
    model = load_model(args.detector_config, args.weights)
    
    # 处理每张图片
    print(f"\n开始处理 {len(IMAGE_LIST)} 张图片...")
    print("方法: 最后一层attention, CLS→patches, heads平均\n")
    
    for i, item in enumerate(IMAGE_LIST, 1):
        img_path = item["path"]
        label = item["label"]
        
        print(f"[{i}/{len(IMAGE_LIST)}] {label}")
        print(f"     路径: {img_path}")
        
        if not os.path.exists(img_path):
            print(f"     ⚠ 图片不存在，跳过")
            continue
        
        try:
            # 预处理
            img, tensor = preprocess_image(img_path)
            tensor = tensor.to(device)
            
            # 获取最后一层注意力
            attention_map = get_last_layer_attention(model, tensor)
            
            # 保存热力图
            save_path = os.path.join(args.output_dir, f"{label}_attention.png")
            visualize_and_save(attention_map, img, save_path, title=label)
            print(f"     ✓ 已保存: {save_path}")
            
            # 模型推理获取预测结果
            with torch.no_grad():
                data_dict = {'image': tensor, 'label': torch.tensor([0]).to(device)}
                pred_dict = model(data_dict, inference=True)
                prob = pred_dict['prob'].cpu().numpy()[0]
                pred_label = "Fake" if prob > 0.5 else "Real"
                print(f"     预测: {pred_label} (prob={prob:.4f})")
            
        except Exception as e:
            print(f"     ✗ 处理失败: {e}")
        
        print()
    
    print("="*60)
    print(f"所有热力图已保存到: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
