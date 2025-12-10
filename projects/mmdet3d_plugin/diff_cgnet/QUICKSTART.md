# 🚀 快速开始指南

## 第一天：环境验证

### 1. 验证CGNet环境

```bash
cd /home/subobo/ro/e2e/CGNet

# 测试原版CGNet是否能跑
bash tools/dist_train.sh \
    projects/mmdet3d_plugin/cgnet/configs/cgnet_r50_nusc.py \
    1 \
    --work-dir work_dirs/test_cgnet
```

**预期结果**: 能正常启动训练，无报错

---

## 第二天：生成锚点

### 2. 生成K-Means聚类锚点

```bash
# 生成锚点
python tools/generate_anchors.py \
    --data-root data/nuscenes \
    --num-clusters 50 \
    --degree 3 \
    --output work_dirs/kmeans_anchors.pth \
    --visualize
```

**检查清单**:
- [ ] `work_dirs/kmeans_anchors.pth` 文件生成
- [ ] `work_dirs/anchors_visualization.png` 图片生成
- [ ] 可视化图中的锚点看起来像车道线（直行、左转、右转）

---

## 第三天：单元测试

### 3. 测试核心模块

```bash
cd projects/mmdet3d_plugin/diff_cgnet/tests

# 测试贝塞尔插值
python -c "
from diff_cgnet.modules.utils import fit_bezier, bezier_interpolate
import numpy as np
import torch

# 测试拟合
points = np.array([[0,0], [1,1], [2,1], [3,0]])
ctrl = fit_bezier(points, n_control=4)
print('✅ 贝塞尔拟合成功:', ctrl.shape)

# 测试插值
ctrl_tensor = torch.from_numpy(ctrl).float().unsqueeze(0)
interp = bezier_interpolate(ctrl_tensor, num_points=20)
print('✅ 贝塞尔插值成功:', interp.shape)
"

# 测试扩散模块
python -c "
from diff_cgnet.modules.diffusion import ColdDiffusion
import torch

diffusion = ColdDiffusion(num_timesteps=1000)
x0 = torch.randn(2, 10, 4, 2)
t = torch.tensor([100, 200])
anchors = torch.randn(10, 4, 2)

xt = diffusion.q_sample(x0, t, anchors)
print('✅ 扩散模块成功:', xt.shape)
"
```

---

## 第四-七天：实现核心模块

### 4. 需要实现的文件（按顺序）

```bash
Day 4: 
  ☐ dense_heads/diff_head.py (扩散检测头)
  
Day 5:
  ☐ detectors/diff_cgnet.py (主检测器)
  
Day 6:
  ☐ 配置文件 configs/diff_cgnet/diff_cgnet_r50_nusc.py
  
Day 7:
  ☐ 集成测试
```

---

## Sanity Check（第8天）

### 5. 过拟合测试

```bash
# 修改配置，只用1个样本
# 在配置文件中设置: data.train.ann_file = 'mini_train.pkl'

# 训练1000步
python tools/train.py \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py \
    --work-dir work_dirs/sanity_check \
    --cfg-options \
    total_epochs=1 \
    data.samples_per_gpu=1
```

**成功标准**:
- [ ] Loss降到 < 0.01
- [ ] 无梯度NaN/Inf
- [ ] 可视化结果与GT重合

---

## 常见问题

### Q1: 锚点生成失败
**原因**: 数据路径不对
**解决**: 检查`data/nuscenes/nuscenes_infos_train.pkl`是否存在

### Q2: 梯度NaN
**原因**: 坐标未归一化或除零
**解决**: 检查`normalize_coords`是否正确调用

### Q3: 匹配器报错
**原因**: 维度不匹配
**解决**: 打印pred_ctrl和gt_ctrl的shape

---

## 下一步

实现剩余的核心模块，然后运行Sanity Check！
