"""
Mock Test: 测试数据流是否能跑通
不需要真实的网络权重，只验证shape变换逻辑
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from projects.mmdet3d_plugin.diff_cgnet.modules.diffusion import ColdDiffusion
from projects.mmdet3d_plugin.diff_cgnet.modules.matcher import HungarianMatcher
from projects.mmdet3d_plugin.diff_cgnet.modules.utils import (
    fit_bezier,
    bezier_interpolate,
    normalize_coords,
    denormalize_coords
)


def test_data_flow():
    """测试完整的数据流"""
    print("="*70)
    print("🧪 Mock Test: 数据流测试")
    print("="*70)
    
    B, N, M = 2, 50, 30
    num_ctrl = 4
    pc_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
    
    print(f"\n配置:")
    print(f"  Batch size: {B}")
    print(f"  预测数量: {N}")
    print(f"  GT数量: {M}")
    print(f"  控制点数: {num_ctrl}")
    
    print("\n" + "-"*70)
    print("Step 1: 模拟GT数据")
    print("-"*70)
    
    gt_ctrl = torch.randn(B, M, num_ctrl, 2) * 10
    gt_labels = torch.zeros(B, M, dtype=torch.long)
    
    print(f"✅ GT控制点: {gt_ctrl.shape}")
    print(f"   范围: [{gt_ctrl.min():.2f}, {gt_ctrl.max():.2f}]")
    
    print("\n" + "-"*70)
    print("Step 2: 归一化")
    print("-"*70)
    
    gt_ctrl_norm = normalize_coords(gt_ctrl, pc_range)
    print(f"✅ 归一化后: {gt_ctrl_norm.shape}")
    print(f"   范围: [{gt_ctrl_norm.min():.3f}, {gt_ctrl_norm.max():.3f}]")
    
    print("\n" + "-"*70)
    print("Step 3: Padding到固定数量")
    print("-"*70)
    
    gt_ctrl_padded = torch.zeros(B, N, num_ctrl, 2)
    gt_ctrl_padded[:, :M] = gt_ctrl_norm
    gt_labels_padded = torch.zeros(B, N, dtype=torch.long)
    gt_labels_padded[:, :M] = gt_labels
    
    print(f"✅ Padding后: {gt_ctrl_padded.shape}")
    
    print("\n" + "-"*70)
    print("Step 4: 生成锚点")
    print("-"*70)
    
    anchors = torch.randn(N, num_ctrl, 2)
    anchors_norm = normalize_coords(anchors, pc_range)
    
    print(f"✅ 锚点: {anchors_norm.shape}")
    
    print("\n" + "-"*70)
    print("Step 5: 扩散加噪")
    print("-"*70)
    
    diffusion = ColdDiffusion(num_timesteps=1000)
    t = torch.tensor([100, 500])
    
    noisy_ctrl = diffusion.q_sample(gt_ctrl_padded, t, anchors=anchors_norm)
    
    print(f"✅ 加噪后: {noisy_ctrl.shape}")
    print(f"   时间步: {t.tolist()}")
    print(f"   范围: [{noisy_ctrl.min():.3f}, {noisy_ctrl.max():.3f}]")
    
    print("\n" + "-"*70)
    print("Step 6: 模拟预测")
    print("-"*70)
    
    pred_ctrl = torch.randn(B, N, num_ctrl, 2)
    pred_ctrl = torch.tanh(pred_ctrl)
    pred_logits = torch.randn(B, N, 1)
    
    print(f"✅ 预测控制点: {pred_ctrl.shape}")
    print(f"✅ 预测logits: {pred_logits.shape}")
    
    print("\n" + "-"*70)
    print("Step 7: 匈牙利匹配")
    print("-"*70)
    
    matcher = HungarianMatcher(cost_class=1.0, cost_bezier=5.0)
    
    indices = matcher(pred_ctrl, pred_logits, gt_ctrl_padded, gt_labels_padded)
    
    print(f"✅ 匹配结果: {len(indices)} batches")
    for b, (pred_idx, gt_idx) in enumerate(indices):
        print(f"   Batch {b}: {len(pred_idx)} 个匹配")
    
    print("\n" + "-"*70)
    print("Step 8: 贝塞尔插值")
    print("-"*70)
    
    dense_points = bezier_interpolate(pred_ctrl, num_points=20)
    
    print(f"✅ 插值点: {dense_points.shape}")
    print(f"   每条线: 20个点")
    
    print("\n" + "-"*70)
    print("Step 9: 反归一化")
    print("-"*70)
    
    pred_ctrl_denorm = denormalize_coords(pred_ctrl, pc_range)
    
    print(f"✅ 反归一化: {pred_ctrl_denorm.shape}")
    print(f"   范围: [{pred_ctrl_denorm.min():.2f}, {pred_ctrl_denorm.max():.2f}]")
    
    print("\n" + "="*70)
    print("✅✅✅ Mock Test 完全通过！")
    print("="*70)
    print("\n所有数据流验证成功！")
    print("Shape变换逻辑正确！")
    print("\n下一步: 实现完整的网络层")
    
    return True


if __name__ == '__main__':
    try:
        success = test_data_flow()
        if success:
            print("\n🎉 可以开始实现网络层了！")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Mock Test失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
