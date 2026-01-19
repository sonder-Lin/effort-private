# Effort模型注意力热力图生成指南

## 📌 概述

本脚本用于生成基于Effort模型的注意力热力图，帮助可视化模型在判断真假时关注的图像区域。

## 🔧 依赖安装

确保你的环境中安装了以下依赖：

```bash
pip install torch torchvision transformers matplotlib opencv-python pillow pyyaml numpy scikit-learn loralib imutils dlib
```

## 📁 文件说明

- `generate_attention_heatmap_simple.py` - **推荐使用**，专门为你的6张图片定制
- `generate_attention_heatmap.py` - 通用版本，可扩展

## 🚀 使用方法

### 方法1：使用简化版脚本（推荐）

```bash
cd /path/to/Effort-AIGI-Detection/DeepfakeBench/training

python generate_attention_heatmap_simple.py \
    --weights /path/to/your/effort_weights.pth \
    --output_dir ./attention_heatmaps \
    --head_fusion mean
```

### 方法2：使用通用版脚本

```bash
python generate_attention_heatmap.py \
    --detector_config config/detector/effort.yaml \
    --weights /path/to/your/effort_weights.pth \
    --output_dir ./attention_heatmaps \
    --method rollout \
    --head_fusion mean
```

## 📊 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weights` | effort模型权重文件路径 | **必填** |
| `--output_dir` | 热力图输出目录 | `./attention_heatmaps` |
| `--detector_config` | 检测器配置文件 | `config/detector/effort.yaml` |
| `--head_fusion` | 多头注意力融合方式: `mean` 或 `max` | `mean` |
| `--method` | 注意力计算方法: `rollout` 或 `last_layer` | `rollout` |

## 📷 预设的6张图片

脚本已内置以下6张图片的路径：

| 图片描述 | 标签 | 路径 |
|----------|------|------|
| Asian-Male [F] | Asian-Male_Fake | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5460_71.png` |
| White-Male [F] | White-Male_Fake | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7184_271.png` |
| Black-Female [F] 0_6477 | Black-Female_Fake_6477 | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6477_221.png` |
| Asian-Female [R] | Asian-Female_Real | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_6710_211.png` |
| White-Female [F] | White-Female_Fake | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_5447_81.png` |
| Black-Female [F] 0_7430 | Black-Female_Fake_7430 | `/home/work/juice/juiceMnt/sonder-zhen/fairness/dfdc/crop_img/0_7430_201.png` |

## 📤 输出说明

对于每张图片，脚本会生成以下热力图：

1. **`{label}_rollout.png`** - Attention Rollout方法生成的热力图（推荐用于论文）
2. **`{label}_last_layer.png`** - 最后一层注意力热力图

每张图包含3个部分：
- 原始图像
- 注意力热力图
- 叠加图

## 🔬 注意力可视化方法说明

### Attention Rollout
- 考虑所有层的注意力传播
- 更能反映模型的整体决策过程
- 包含残差连接的影响
- **推荐用于论文可视化**

### Last Layer Attention
- 仅使用最后一层的注意力权重
- 反映模型最终阶段关注的区域
- 计算更快

## 🛠 如需修改图片列表

编辑 `generate_attention_heatmap_simple.py` 文件中的 `IMAGE_LIST` 变量：

```python
IMAGE_LIST = [
    {
        "path": "/your/image/path.png",
        "label": "your_label"
    },
    # ... 添加更多图片
]
```

## ⚠️ 注意事项

1. 确保CLIP模型文件夹存在于正确位置（参见 `effort_detector.py` 中的路径）
2. 如果GPU内存不足，可以在脚本开头将device改为CPU
3. 输出图像分辨率可通过修改 `plt.savefig(... dpi=200 ...)` 调整

## 📞 示例完整命令

```bash
# 在服务器上运行
cd /home/work/juice/juiceMnt/sonder-zhen/Effort-AIGI-Detection/DeepfakeBench/training

python generate_attention_heatmap_simple.py \
    --weights /home/work/juice/juiceMnt/sonder-zhen/weights/effort_dfdc.pth \
    --output_dir /home/work/juice/juiceMnt/sonder-zhen/fairness/attention_heatmaps
```

将 `effort_dfdc.pth` 替换为你实际的权重文件路径。

