# Flow Matching + GRPO for Centerline Generation

替代原有的Cold Diffusion方案，采用更高效的Flow Matching + GRPO强化学习。

## 🎯 核心改进

| 对比 | 原方案 (Cold Diffusion) | 新方案 (Flow Matching + GRPO) |
|------|------------------------|------------------------------|
| 扩散类型 | Cold Diffusion (退化到anchor) | Flow Matching (OT-CFM) |
| 采样步数 | 4步 DDIM | **1步** Euler |
| 训练目标 | 预测x0 | 预测速度场v |
| 拓扑学习 | BCE | BCE + **GRPO优化** |
| 复杂度 | 高 (JAQ/BSC/GNN/Teacher Forcing) | **低** (简洁设计) |

## 📁 新增文件

```
diff_cgnet/
├── modules/
│   ├── flow_matching.py     # Flow Matching核心 ✨
│   └── grpo.py              # GRPO强化学习 ✨
├── dense_heads/
│   └── flow_head.py         # Flow检测头 ✨
├── detectors/
│   └── flow_cgnet.py        # 主检测器 ✨
└── FLOW_MATCHING_README.md  # 本文档 ✨
```

## 🚀 快速开始

### 训练

```bash
bash tools/dist_train.sh \
    projects/configs/flow_cgnet/flow_cgnet_r50_nusc.py 8
```

### 测试

```bash
python tools/test.py \
    projects/configs/flow_cgnet/flow_cgnet_r50_nusc.py \
    work_dirs/flow_cgnet_r50_nusc/latest.pth \
    --eval bbox
```

## 🔧 核心模块说明

### 1. FlowMatchingModule (`modules/flow_matching.py`)

```python
# 训练: 学习速度场
t = torch.rand(B)  # 采样时间
x_t = (1-t)*noise + t*target  # 插值
v_true = target - noise  # 真实速度
v_pred = model(x_t, t, condition)  # 预测速度
loss = MSE(v_pred, v_true)

# 推理: 单步采样
x_1 = noise + model(noise, t=0, condition)
```

### 2. GRPO (`modules/grpo.py`)

```python
# 采样K个候选
candidates = [sample() for _ in range(K)]

# 计算reward
rewards = compute_reward(candidates, gt)

# 组内相对优势
advantages = rewards - rewards.mean()

# 策略梯度
loss = -(advantages * log_prob).mean()
```

### 3. FlowCenterlineHead (`dense_heads/flow_head.py`)

- **Flow Matching**: 生成Bézier控制点
- **TopologyHead**: 预测邻接矩阵 (利用端点距离先验)
- **GRPO**: epoch 12后启用，优化整体质量

## 📊 训练策略

| 阶段 | Epochs | 内容 |
|------|--------|------|
| Stage 1 | 1-12 | Flow Matching + BCE拓扑 (监督学习) |
| Stage 2 | 13-24 | + GRPO微调 (强化学习) |

## 🔑 关键设计

### 借鉴GoalFlow
- 单步采样
- 简洁的MLP速度网络

### 借鉴DIVER
- AdaLN时间调制 (scale + shift)
- GRPO组内相对优势
- Mish激活函数

### 自创设计
- 端点距离作为拓扑先验
- 多维度Reward (Chamfer + Topo F1 + Smoothness + Diversity)

## ⚠️ 注意事项

1. **坐标归一化**: 所有坐标在[0,1]范围内
2. **GRPO启动时机**: 建议在模型收敛后启用 (epoch 12+)
3. **采样步数**: 默认1步，可增加到2-4步提升精度

## 📈 预期效果

- 推理速度: 比原方案快 **4倍+**
- 几何精度: 与原方案持平或更好
- 拓扑精度: GRPO优化后提升 **2-5%**
