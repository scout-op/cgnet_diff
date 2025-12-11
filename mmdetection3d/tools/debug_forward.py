"""
Debug脚本：单步调试forward_train
确保所有维度对齐，无NaN/Inf
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'projects/mmdet3d_plugin')

from diff_cgnet.dense_heads.diff_head import DiffusionCenterlineHead
from diff_cgnet.modules.utils import fit_bezier
import numpy as np


def test_forward_train():
    """测试forward_train的完整流程"""
    
    print("="*70)
    print("🧪 Debug Forward Train")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    B, N = 2, 50
    H, W = 200, 100
    C = 256
    
    print(f"\nConfig:")
    print(f"  Batch: {B}")
    print(f"  Num queries: {N}")
    print(f"  BEV size: ({H}, {W})")
    
    print("\n" + "-"*70)
    print("Step 1: 创建模型")
    print("-"*70)
    
    head = DiffusionCenterlineHead(
        num_classes=1,
        embed_dims=C,
        num_queries=N,
        num_ctrl_points=4,
        use_gnn=True,
        use_jaq=False,
        use_bsc=False,
        with_multiview_supervision=True
    ).to(device)
    
    head.load_anchors('work_dirs/kmeans_anchors.pth')
    
    print("✅ 模型创建成功")
    print(f"   Anchors shape: {head.anchors.shape if head.anchors is not None else 'None'}")
    
    print("\n" + "-"*70)
    print("Step 2: 构造假数据")
    print("-"*70)
    
    bev_features = torch.randn(B, C, H, W, device=device)
    print(f"✅ BEV features: {bev_features.shape}")
    
    class FakeGT:
        def __init__(self, instance_list):
            self.instance_list = instance_list
    
    gt_bboxes_list = []
    gt_labels_list = []
    
    for b in range(B):
        num_gt = np.random.randint(5, 15)
        
        instance_list = []
        for _ in range(num_gt):
            line = np.random.randn(20, 2) * 5
            instance_list.append(line)
        
        gt_bboxes_list.append(FakeGT(instance_list))
        gt_labels_list.append(torch.zeros(num_gt, dtype=torch.long, device=device))
    
    print(f"✅ GT data created")
    print(f"   Batch 0: {len(gt_bboxes_list[0].instance_list)} lines")
    print(f"   Batch 1: {len(gt_bboxes_list[1].instance_list)} lines")
    
    print("\n" + "-"*70)
    print("Step 3: 测试prepare_gt")
    print("-"*70)
    
    try:
        targets, labels, mask = head.prepare_gt(
            gt_bboxes_list, gt_labels_list, device
        )
        
        print(f"✅ prepare_gt成功")
        print(f"   Targets: {targets.shape}")
        print(f"   Labels: {labels.shape}")
        print(f"   Mask: {mask.shape}")
        print(f"   Positive samples: {mask.sum(dim=1).tolist()}")
        
    except Exception as e:
        print(f"❌ prepare_gt失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*70)
    print("Step 4: 测试扩散加噪")
    print("-"*70)
    
    try:
        t = torch.randint(0, 1000, (B,), device=device)
        
        if head.anchors is not None:
            anchors = head.anchors.to(device)
            if anchors.dim() == 3:
                anchors = anchors.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            anchors = None
        
        noisy_ctrl = head.diffusion.q_sample(targets, t, anchors=anchors)
        
        print(f"✅ 扩散加噪成功")
        print(f"   Noisy ctrl: {noisy_ctrl.shape}")
        print(f"   Time steps: {t.tolist()}")
        print(f"   Value range: [{noisy_ctrl.min():.3f}, {noisy_ctrl.max():.3f}]")
        
        if torch.isnan(noisy_ctrl).any():
            print("❌ 包含NaN!")
            return False
        if torch.isinf(noisy_ctrl).any():
            print("❌ 包含Inf!")
            return False
            
    except Exception as e:
        print(f"❌ 扩散加噪失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*70)
    print("Step 5: 测试forward_single_step")
    print("-"*70)
    
    try:
        outputs = head.forward_single_step(
            noisy_ctrl, bev_features, t, self_cond=None
        )
        
        if head.with_multiview_supervision:
            all_pred_ctrl, all_pred_logits, all_features = outputs
            print(f"✅ forward_single_step成功（多层）")
            print(f"   Num layers: {len(all_pred_ctrl)}")
            print(f"   Pred ctrl shape: {all_pred_ctrl[-1].shape}")
            print(f"   Pred logits shape: {all_pred_logits[-1].shape}")
        else:
            pred_ctrl, pred_logits, features = outputs
            print(f"✅ forward_single_step成功（单层）")
            print(f"   Pred ctrl: {pred_ctrl.shape}")
            print(f"   Pred logits: {pred_logits.shape}")
            
    except Exception as e:
        print(f"❌ forward_single_step失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*70)
    print("Step 6: 测试完整forward_train")
    print("-"*70)
    
    try:
        head.train()
        
        losses = head.forward_train(
            bev_features,
            gt_bboxes_list,
            gt_labels_list,
            img_metas=[{} for _ in range(B)],
            epoch=0
        )
        
        print(f"✅ forward_train成功")
        print(f"   Losses: {list(losses.keys())}")
        for k, v in losses.items():
            print(f"   {k}: {v.item():.4f}")
        
        total_loss = sum(losses.values())
        print(f"   Total loss: {total_loss.item():.4f}")
        
        if torch.isnan(total_loss):
            print("❌ Loss是NaN!")
            return False
            
    except Exception as e:
        print(f"❌ forward_train失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-"*70)
    print("Step 7: 测试反向传播")
    print("-"*70)
    
    try:
        total_loss.backward()
        
        has_nan = False
        for name, param in head.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN梯度: {name}")
                    has_nan = True
        
        if not has_nan:
            print("✅ 反向传播成功，无NaN梯度")
        else:
            return False
            
    except Exception as e:
        print(f"❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅✅✅ 所有测试通过！")
    print("="*70)
    print("\n代码逻辑正确，可以开始训练！")
    
    return True


if __name__ == '__main__':
    success = test_forward_train()
    sys.exit(0 if success else 1)
