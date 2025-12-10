# 🎯 下一步行动清单

## ✅ 已完成（今天）

1. ✅ 实现所有基础模块
2. ✅ 修复关键bug（锚点匹配、坐标clamp）
3. ✅ 推送到GitHub
4. ✅ Code Review并改进

**GitHub**: https://github.com/scout-op/cgnet_diff/tree/diffusion-implementation

---

## 🚀 立即执行（现在，5分钟）

### Step 1: Mock Test

```bash
cd /home/subobo/ro/e2e/CGNet
python projects/mmdet3d_plugin/diff_cgnet/tests/test_mock.py
```

**目标**: 验证数据流，所有shape变换正确

**预期输出**:
```
✅ GT控制点: torch.Size([2, 30, 4, 2])
✅ 归一化后: torch.Size([2, 30, 4, 2])
✅ Padding后: torch.Size([2, 50, 4, 2])
✅ 加噪后: torch.Size([2, 50, 4, 2])
✅ 匹配结果: 2 batches
✅ Mock Test 完全通过！
```

---

## 📝 明天的任务（2-3小时）

### Task 1: 实现简单版本的网络层

在 `dense_heads/diff_head.py` 中：

```python
def _init_layers(self):
    """先用最简单的MLP，验证训练流程"""
    
    # 简单的编码器
    self.ctrl_encoder = nn.Sequential(
        nn.Linear(8, 256),
        nn.ReLU(),
        nn.Linear(256, 256)
    )
    
    # 简单的解码器
    self.ctrl_decoder = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 8)
    )
    
    # 分类头
    self.cls_head = nn.Linear(256, 1)
```

### Task 2: 实现简单的forward

```python
def forward_single_step(self, noisy_ctrl, bev_features, t):
    """简化版本，先跑通"""
    B, N = noisy_ctrl.shape[:2]
    
    # 编码
    ctrl_flat = noisy_ctrl.flatten(2)
    features = self.ctrl_encoder(ctrl_flat)
    
    # 解码
    pred_flat = self.ctrl_decoder(features)
    pred_ctrl = pred_flat.view(B, N, 4, 2)
    pred_ctrl = torch.tanh(pred_ctrl)
    
    # 分类
    pred_logits = self.cls_head(features)
    
    return pred_ctrl, pred_logits, features
```

### Task 3: Overfit Test

```bash
# 创建mini数据集
python -c "
import pickle
data = pickle.load(open('data/nuscenes/nuscenes_infos_temporal_train.pkl', 'rb'))
mini = {'infos': data['infos'][:1], 'metadata': data.get('metadata', {})}
pickle.dump(mini, open('data/nuscenes/mini_train.pkl', 'wb'))
"

# 修改配置
# ann_file = 'data/nuscenes/mini_train.pkl'

# 训练
python tools/train.py configs/diff_cgnet/diff_cgnet_r50_nusc.py
```

**成功标准**: Loss < 0.01

---

## 📅 本周计划

### 今天（Day 1）
- [x] 基础模块实现
- [x] Bug修复
- [x] 推送到GitHub
- [ ] Mock Test ← **现在做这个**

### 明天（Day 2）
- [ ] 实现简单网络层
- [ ] Overfit Test

### 后天（Day 3）
- [ ] 实现完整网络层
- [ ] 添加Transformer Decoder
- [ ] 添加Deformable Attention

### Day 4-5
- [ ] 小规模训练（10%数据）
- [ ] 调整超参数

### Day 6-7
- [ ] 全量训练
- [ ] 评估指标

---

## ⚠️ 重要提醒

### 不要做的事

❌ 不要一次性写完所有代码
❌ 不要跳过Mock Test
❌ 不要跳过Overfit Test
❌ 不要在Overfit Test失败时继续

### 要做的事

✅ 严格按照Phase执行
✅ 每个Phase都要验证
✅ 遇到问题立即Debug
✅ 保持代码整洁

---

## 🎯 当前状态

```
项目进度: ████████░░ 80%

已完成:
✅ 基础模块
✅ Bug修复
✅ 推送GitHub

待完成:
🔄 Mock Test (5分钟)
⏰ 网络实现 (2小时)
⏰ Overfit Test (1小时)
⏰ 完整训练 (2-3天)
```

---

## 🚀 立即行动

**现在就运行:**

```bash
python projects/mmdet3d_plugin/diff_cgnet/tests/test_mock.py
```

**如果通过，明天开始实现网络层！**

**稳扎稳打，一步一个脚印！** 🎯
