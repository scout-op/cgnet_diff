# 🎯 增强模块使用指南

## ✅ 已实现的增强模块

所有增强模块已实现完毕！可以根据需要启用。

---

## 📦 模块清单

### **1. JAQ - Junction Aware Query** ✅

**功能**: 增强路口感知能力

**文件**: `modules/jaq.py`

**预期提升**: +1-2% GEO F1, +1% TOPO F1

**启用方式**:
```python
# 在配置文件中
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_jaq=True,  # ← 启用JAQ
    dilate_radius=9,  # 路口膨胀半径
    ...
)
```

**工作原理**:
1. 从BEV特征解码路口特征
2. 生成路口热图
3. 使用线性注意力增强query
4. 提供路口位置先验

---

### **2. BSC - Bézier Space Connection** ✅

**功能**: 贝塞尔空间连续性约束

**文件**: `modules/bsc.py`

**预期提升**: +0.5-1% APLS（连续性）

**启用方式**:
```python
# 在配置文件中
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_bsc=True,  # ← 启用BSC
    ...
)
```

**工作原理**:
1. 找到连接的线对
2. 在贝塞尔空间中融合特征
3. 预测连接处的控制点
4. 施加连续性约束

---

### **3. 评估工具** ✅

**功能**: 计算所有评估指标

**文件**: `evaluation/centerline_metrics.py`, `tools/eval_diff_cgnet.py`

**使用方式**:
```bash
python tools/eval_diff_cgnet.py \
    --results work_dirs/diff_cgnet/results.pkl \
    --gt-file data/nuscenes/anns/gt_centerlines.pkl \
    --thresholds 0.5 1.0 1.5
```

**输出指标**:
- GEO F1
- TOPO F1
- APLS
- Chamfer Distance

---

### **4. 可视化工具** ✅

**功能**: 可视化预测结果

**文件**: `tools/visualize_diff_cgnet.py`

**使用方式**:
```bash
python tools/visualize_diff_cgnet.py \
    --results work_dirs/diff_cgnet/results.pkl \
    --gt-file data/nuscenes/anns/gt_centerlines.pkl \
    --output-dir work_dirs/visualizations \
    --num-samples 50 \
    --show-topology \
    --save-video
```

**输出**:
- 对比图（GT vs Pred）
- 拓扑连接可视化
- 视频（可选）

---

## 🎯 使用策略

### **策略1: 基础版本（推荐先跑）**

```python
# configs/diff_cgnet/diff_cgnet_r50_nusc.py

pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_gnn=True,   # ✅ 使用GNN
    use_jaq=False,  # ❌ 暂不使用
    use_bsc=False,  # ❌ 暂不使用
    ...
)
```

**优势**: 
- 简单，易调试
- 训练更快
- 先验证扩散模型本身的效果

---

### **策略2: 添加JAQ（提升路口）**

```python
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_gnn=True,
    use_jaq=True,   # ✅ 启用JAQ
    use_bsc=False,
    dilate_radius=9,
    ...
)
```

**何时使用**: 基础版本训练后，如果路口精度不够

**预期**: +1-2% mAP

---

### **策略3: 添加BSC（提升连续性）**

```python
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_gnn=True,
    use_jaq=True,
    use_bsc=True,   # ✅ 启用BSC
    ...
)
```

**何时使用**: 如果连续性指标（APLS）不够好

**预期**: +0.5-1% APLS

---

### **策略4: 全功能版本**

```python
pts_bbox_head=dict(
    type='DiffusionCenterlineHead',
    use_gnn=True,   # ✅ 拓扑预测
    use_jaq=True,   # ✅ 路口增强
    use_bsc=True,   # ✅ 连续性约束
    dilate_radius=9,
    ...
)
```

**何时使用**: 最终版本，追求最佳性能

**预期**: 所有指标最优

---

## 📊 性能预期

| 版本 | GEO F1 | TOPO F1 | APLS | 说明 |
|------|--------|---------|------|------|
| 基础版 | 55-56 | 43-44 | 31-32 | 扩散+GNN |
| +JAQ | 56-57 | 44-45 | 32-33 | +路口增强 |
| +BSC | 56-57 | 44-45 | 33-34 | +连续性 |
| **全功能** | **57-58** | **45-46** | **33-34** | **最佳** |

**CGNet baseline**: 54.7 / 42.2 / 30.7

---

## 🚀 实施建议

### **第1周**: 基础版本
```bash
# 不启用JAQ和BSC
use_jaq=False
use_bsc=False

# 训练
bash tools/train_diff_cgnet.sh configs/diff_cgnet/diff_cgnet_r50_nusc.py 8

# 评估
python tools/eval_diff_cgnet.py --results ... --gt-file ...

# 可视化
python tools/visualize_diff_cgnet.py --results ... --gt-file ...
```

### **第2周**: 添加JAQ
```bash
# 启用JAQ
use_jaq=True

# 重新训练
# 对比性能
```

### **第3周**: 添加BSC
```bash
# 启用BSC
use_bsc=True

# 最终训练
# 论文实验
```

---

## 🔧 调试技巧

### **如果JAQ导致训练不稳定**:
```python
# 降低junction loss权重
loss_weights = {
    'geometry': 5.0,
    'topology': 1.0,
    'junction': 0.05,  # ← 从0.1降到0.05
}
```

### **如果BSC导致过拟合**:
```python
# 降低BSC loss权重
loss_weights = {
    'bsc': 0.05,  # ← 从0.1降到0.05
}
```

---

## 📝 模块状态

```
✅ JAQ模块: 已实现，可启用
✅ BSC模块: 已实现，可启用
✅ 评估工具: 已实现，可使用
✅ 可视化: 已实现，可使用

默认状态: 全部禁用
建议: 渐进式启用
```

---

**所有增强模块已实现，可以根据需要灵活启用！** 🎯
