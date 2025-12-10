# 🎯 渐进式开发指南

基于Code Review的建议，采用**渐进式开发策略**，避免一次性写完导致的挫败感。

---

## 📋 开发顺序（严格按此执行）

### ✅ Phase 0: 基础验证（已完成）

- [x] 创建项目结构
- [x] 实现基础模块
- [x] 修复关键bug（锚点匹配、坐标clamp）

---

### 🔄 Phase 1: Mock Test（今天，30分钟）

**目标**: 验证数据流，不需要真实网络

```bash
# 运行Mock Test
python projects/mmdet3d_plugin/diff_cgnet/tests/test_mock.py
```

**检查项**:
- [ ] 所有shape变换正确
- [ ] 归一化/反归一化正确
- [ ] 匹配逻辑正确
- [ ] 无维度错误

**如果失败**: Debug，不要继续

---

### 🔄 Phase 2: 网络骨架（明天，2小时）

**目标**: 实现网络层，但先用简单的placeholder

#### 修改 `dense_heads/diff_head.py`

```python
# 先实现最简单的版本
class DiffusionCenterlineHead(nn.Module):
    def forward_single_step(self, noisy_ctrl, bev_features, t):
        """暂时用简单的MLP"""
        B, N = noisy_ctrl.shape[:2]
        
        # Placeholder: 简单的MLP
        ctrl_flat = noisy_ctrl.flatten(2)  # [B, N, 8]
        
        # 简单的线性层
        pred = self.simple_mlp(ctrl_flat)  # [B, N, 8]
        pred_ctrl = pred.view(B, N, 4, 2)
        pred_ctrl = torch.tanh(pred_ctrl)
        
        pred_logits = torch.zeros(B, N, 1)
        features = torch.zeros(B, N, 256)
        
        return pred_ctrl, pred_logits, features
```

**测试**:
```bash
# 运行forward一次，看是否报错
python -c "
from diff_cgnet.dense_heads.diff_head import DiffusionCenterlineHead
import torch

head = DiffusionCenterlineHead()
noisy = torch.randn(2, 50, 4, 2)
bev = torch.randn(2, 256, 100, 50)
t = torch.tensor([100, 200])

pred_ctrl, pred_logits, feat = head.forward_single_step(noisy, bev, t)
print(f'✅ Forward成功: {pred_ctrl.shape}')
"
```

---

### 🔄 Phase 3: Overfit Test（后天，1小时）

**目标**: 在1个样本上完美过拟合

```bash
# 修改配置，只用1个样本
# 创建mini数据集
python -c "
import pickle
data = pickle.load(open('data/nuscenes/nuscenes_infos_temporal_train.pkl', 'rb'))
mini_data = {'infos': data['infos'][:1]}
pickle.dump(mini_data, open('data/nuscenes/mini_train.pkl', 'wb'))
print('✅ Mini数据集创建成功')
"

# 修改配置文件
# ann_file = 'data/nuscenes/mini_train.pkl'

# 训练1000步
python tools/train.py \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py \
    --work-dir work_dirs/overfit_test
```

**成功标准**:
- [ ] Loss降到 < 0.01
- [ ] 无梯度NaN/Inf
- [ ] 训练稳定

**如果失败**: 
1. 检查匹配逻辑
2. 检查损失计算
3. 检查坐标归一化

---

### 🔄 Phase 4: 完整实现（第4-5天）

**目标**: 实现完整的网络层

#### 添加真实的网络组件

```python
# 1. 时间嵌入
self.time_mlp = nn.Sequential(...)

# 2. Transformer Decoder
self.transformer_decoder = nn.TransformerDecoder(...)

# 3. Deformable Attention
self.deform_attn = BezierDeformableAttention(...)

# 4. 预测头
self.ctrl_head = nn.Sequential(...)
self.cls_head = nn.Sequential(...)
```

**测试**: 重新运行Overfit Test

---

### 🔄 Phase 5: 小规模训练（第6天）

```bash
# 10%数据训练
python tools/train.py \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py \
    --work-dir work_dirs/small_scale
```

**检查**:
- [ ] Loss曲线正常
- [ ] 验证集指标合理
- [ ] 可视化结果正常

---

### 🔄 Phase 6: 全量训练（第7-10天）

```bash
# 8卡训练
bash tools/train_diff_cgnet.sh \
    configs/diff_cgnet/diff_cgnet_r50_nusc.py 8
```

---

## ⚠️ 关键检查点

### Checkpoint 1: Mock Test
**如果失败**: 修复shape问题，不要继续

### Checkpoint 2: Overfit Test
**如果失败**: 修复训练逻辑，不要继续

### Checkpoint 3: 小规模训练
**如果失败**: 调整超参数，不要全量训练

---

## 🔧 Debug技巧

### 1. 打印Shape
```python
# 在每个关键步骤打印
print(f"Debug: noisy_ctrl.shape = {noisy_ctrl.shape}")
print(f"Debug: pred_ctrl.shape = {pred_ctrl.shape}")
```

### 2. 检查数值范围
```python
# 确保数值在合理范围
assert noisy_ctrl.min() >= -2 and noisy_ctrl.max() <= 2, "数值异常"
```

### 3. 梯度检查
```python
# 每100步检查一次
if step % 100 == 0:
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm={param.grad.norm():.4f}")
```

---

## 📊 进度追踪

```
✅ Phase 0: 基础验证 (100%)
🔄 Phase 1: Mock Test (0%)      ← 现在在这里
⏰ Phase 2: 网络骨架 (0%)
⏰ Phase 3: Overfit Test (0%)
⏰ Phase 4: 完整实现 (0%)
⏰ Phase 5: 小规模训练 (0%)
⏰ Phase 6: 全量训练 (0%)
```

---

## 🎯 立即执行

```bash
# 运行Mock Test
python projects/mmdet3d_plugin/diff_cgnet/tests/test_mock.py
```

**预期时间**: 1分钟

**如果通过**: 继续Phase 2

**如果失败**: Debug后重试

---

**按照这个顺序，稳扎稳打！** 🎯
