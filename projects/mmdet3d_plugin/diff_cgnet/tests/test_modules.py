"""
单元测试：验证核心模块的正确性
运行: python -m pytest projects/mmdet3d_plugin/diff_cgnet/tests/test_modules.py -v
或者: python projects/mmdet3d_plugin/diff_cgnet/tests/test_modules.py
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from projects.mmdet3d_plugin.diff_cgnet.modules.utils import (
    fit_bezier,
    bezier_interpolate,
    cubic_bezier_interpolate,
    normalize_coords,
    denormalize_coords,
    chamfer_distance
)
from projects.mmdet3d_plugin.diff_cgnet.modules.diffusion import ColdDiffusion


def test_bezier_fitting():
    """测试贝塞尔拟合"""
    print("\n" + "="*60)
    print("测试1: 贝塞尔拟合")
    print("="*60)
    
    points = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.5],
        [3.0, 0.5]
    ])
    
    ctrl = fit_bezier(points, n_control=4)
    
    assert ctrl.shape == (4, 2), f"控制点形状错误: {ctrl.shape}"
    assert np.allclose(ctrl[0], points[0]), "起点不匹配"
    assert np.allclose(ctrl[-1], points[-1]), "终点不匹配"
    
    print(f"✅ 贝塞尔拟合成功")
    print(f"   输入点数: {len(points)}")
    print(f"   控制点: {ctrl.shape}")
    print(f"   起点: {ctrl[0]}")
    print(f"   终点: {ctrl[-1]}")


def test_bezier_interpolation():
    """测试贝塞尔插值"""
    print("\n" + "="*60)
    print("测试2: 贝塞尔插值")
    print("="*60)
    
    ctrl_points = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 0.0]]
    ], dtype=torch.float32)
    
    points = cubic_bezier_interpolate(ctrl_points, num_points=20)
    
    assert points.shape == (1, 20, 2), f"插值点形状错误: {points.shape}"
    assert torch.allclose(points[0, 0], ctrl_points[0, 0], atol=1e-5), "起点不匹配"
    assert torch.allclose(points[0, -1], ctrl_points[0, -1], atol=1e-5), "终点不匹配"
    
    diffs = points[0, 1:] - points[0, :-1]
    lengths = torch.norm(diffs, dim=-1)
    assert (lengths > 0).all(), "插值点有重复"
    
    print(f"✅ 贝塞尔插值成功")
    print(f"   控制点: {ctrl_points.shape}")
    print(f"   插值点: {points.shape}")
    print(f"   曲线长度: {lengths.sum():.2f}")


def test_coordinate_normalization():
    """测试坐标归一化"""
    print("\n" + "="*60)
    print("测试3: 坐标归一化")
    print("="*60)
    
    pc_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
    
    coords = torch.tensor([
        [[-15.0, -30.0], [0.0, 0.0], [15.0, 30.0]],
        [[-10.0, -20.0], [5.0, 10.0], [10.0, 20.0]]
    ])
    
    normalized = normalize_coords(coords, pc_range)
    
    assert normalized.min() >= 0.01, "归一化最小值错误"
    assert normalized.max() <= 0.99, "归一化最大值错误"
    
    denormalized = denormalize_coords(normalized, pc_range)
    
    assert torch.allclose(denormalized, coords, atol=0.1), "反归一化不匹配"
    
    print(f"✅ 坐标归一化成功")
    print(f"   原始范围: [{coords.min():.1f}, {coords.max():.1f}]")
    print(f"   归一化范围: [{normalized.min():.3f}, {normalized.max():.3f}]")


def test_chamfer_distance():
    """测试Chamfer Distance"""
    print("\n" + "="*60)
    print("测试4: Chamfer Distance")
    print("="*60)
    
    pred = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    ])
    
    gt = torch.tensor([
        [[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]],
    ])
    
    dist = chamfer_distance(pred, gt)
    
    assert dist > 0, "距离应该大于0"
    assert dist < 1.0, "距离过大"
    
    identical_dist = chamfer_distance(pred[:1], pred[:1])
    assert identical_dist < 1e-5, "相同点集距离应该接近0"
    
    print(f"✅ Chamfer Distance成功")
    print(f"   距离: {dist:.4f}")


def test_cold_diffusion():
    """测试Cold Diffusion模块"""
    print("\n" + "="*60)
    print("测试5: Cold Diffusion")
    print("="*60)
    
    diffusion = ColdDiffusion(num_timesteps=1000, beta_schedule='cosine')
    
    B, N = 2, 10
    x0 = torch.randn(B, N, 4, 2)
    t = torch.tensor([100, 500])
    anchors = torch.randn(N, 4, 2)
    
    xt = diffusion.q_sample(x0, t, anchors=anchors)
    
    assert xt.shape == x0.shape, f"扩散输出形状错误: {xt.shape}"
    assert not torch.isnan(xt).any(), "扩散输出包含NaN"
    assert not torch.isinf(xt).any(), "扩散输出包含Inf"
    
    pred_x0 = torch.randn_like(xt)
    xt_prev = diffusion.ddim_sample_step(xt, pred_x0, t[0])
    
    assert xt_prev.shape == xt.shape, "DDIM采样形状错误"
    
    print(f"✅ Cold Diffusion成功")
    print(f"   时间步: {t.tolist()}")
    print(f"   Alpha值: {diffusion.alphas_cumprod[t].tolist()}")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪"*30)
    print("开始单元测试")
    print("🧪"*30)
    
    try:
        test_bezier_fitting()
        test_bezier_interpolation()
        test_coordinate_normalization()
        test_chamfer_distance()
        test_cold_diffusion()
        
        print("\n" + "="*60)
        print("✅✅✅ 所有单元测试通过！")
        print("="*60)
        print("\n下一步: 生成K-Means锚点")
        print("运行: python tools/generate_anchors.py --visualize")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 测试失败: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
