# DiffCGNet: Diffusion-based Centerline Graph Generation

基于CGNet的扩散模型中心线图生成方法

## 🎯 核心创新

1. **贝塞尔空间扩散**: 在8维控制点空间扩散（vs 40维点空间）
2. **Cold Diffusion**: 使用确定性退化，保持几何结构
3. **Teacher Forcing**: 渐进式训练GNN，避免cold start
4. **端到端训练**: 从图像到中心线图，完全可微

## 📁 项目结构

```
diff_cgnet/
├── detectors/          # 主检测器
├── dense_heads/        # 扩散检测头
├── modules/            # 核心模块
│   ├── diffusion.py    # 扩散调度
│   ├── matcher.py      # 匈牙利匹配
│   ├── sampler.py      # 贝塞尔Deformable Attention
│   └── utils.py        # 几何工具
├── hooks/              # 训练钩子
│   └── teacher_forcing.py
└── tests/              # 单元测试
```

## 🚀 快速开始

### Step 1: 生成锚点

```bash
python tools/generate_anchors.py \
    --data-root data/nuscenes \
    --num-clusters 50 \
    --visualize
```

### Step 2: Sanity Check

```bash
# 先在1个样本上过拟合，验证代码逻辑
python projects/mmdet3d_plugin/diff_cgnet/tests/test_sanity_check.py
```

### Step 3: 训练

```bash
bash tools/dist_train.sh \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py \
    8
```

## 📊 实施路线

- [x] Phase 0: 预处理与验证（1-2天）
  - [x] 创建项目结构
  - [x] 实现几何工具
  - [x] 实现扩散模块
  - [x] 实现匹配器
  - [ ] 生成K-Means锚点
  
- [ ] Phase 1: 核心模块（1周）
  - [ ] 实现扩散检测头
  - [ ] 实现去噪网络
  - [ ] 集成到CGNet
  
- [ ] Phase 1.5: Sanity Check
  - [ ] 过拟合测试
  - [ ] Debug
  
- [ ] Phase 2: 全量训练（1-2周）
  - [ ] 小规模训练
  - [ ] 全量训练
  - [ ] 评估指标

## ⚠️ 重要提示

1. **坐标系**: 始终使用`normalize_coords`和`denormalize_coords`
2. **梯度检查**: 每100步检查一次梯度
3. **Sanity Check**: 必须通过才能全量训练

## 📝 下一步

实现`dense_heads/diff_head.py`和`detectors/diff_cgnet.py`
