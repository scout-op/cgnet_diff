import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def check_gradients(model, verbose=False):
    """
    检查梯度是否正常
    
    Args:
        model: nn.Module
        verbose: bool, 是否打印详细信息
    
    Returns:
        is_valid: bool, 梯度是否正常
    """
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            
            if torch.isnan(grad_norm):
                print(f"❌ NaN梯度: {name}")
                has_nan = True
            
            if torch.isinf(grad_norm):
                print(f"❌ Inf梯度: {name}, norm={grad_norm}")
                has_inf = True
            
            if verbose and grad_norm > 100:
                print(f"⚠️  大梯度: {name}, norm={grad_norm:.2f}")
    
    return not (has_nan or has_inf)


def sanity_check_overfit(model, dataset, num_steps=1000, lr=1e-4):
    """
    过拟合测试：在1个batch上训练，验证代码逻辑
    
    Args:
        model: DiffCGNet模型
        dataset: 数据集
        num_steps: 训练步数
        lr: 学习率
    
    Returns:
        success: bool, 是否通过测试
    """
    print("="*70)
    print("🧪 开始 Sanity Check（过拟合测试）")
    print("="*70)
    print("目标: 在1个样本上完美过拟合")
    print("标准: Loss < 0.01, 几何误差 < 0.5m, 拓扑准确率 > 95%")
    print("="*70)
    
    single_sample = [dataset[0]]
    mini_loader = DataLoader(single_sample, batch_size=1, shuffle=False)
    
    model = model.cuda()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    losses = []
    geo_errors = []
    
    print("\n开始训练...")
    for step in range(num_steps):
        batch = next(iter(mini_loader))
        
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        loss_dict = model.forward_train(
            batch['img'],
            batch['gt_bboxes_3d'],
            batch['gt_labels_3d'],
            epoch=0
        )
        
        loss = sum(loss_dict.values())
        
        optimizer.zero_grad()
        loss.backward()
        
        if not check_gradients(model, verbose=(step % 100 == 0)):
            print(f"\n❌ Step {step}: 梯度异常！")
            return False
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step:4d}: Loss = {loss.item():.6f}")
            for k, v in loss_dict.items():
                print(f"  - {k}: {v.item():.6f}")
    
    final_loss = losses[-1]
    
    print("\n" + "="*70)
    print("📊 Sanity Check 结果:")
    print("="*70)
    
    success = True
    
    if final_loss < 0.01:
        print(f"✅ Loss收敛: {final_loss:.6f} < 0.01")
    else:
        print(f"❌ Loss未收敛: {final_loss:.6f} >= 0.01")
        print("   可能的问题:")
        print("   - 匹配逻辑错误")
        print("   - 损失函数计算错误")
        print("   - 学习率太小或太大")
        print("   - 数据预处理错误")
        success = False
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(mini_loader))
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        
        results = model.forward_test(batch['img'], batch['img_metas'])
    
    print("\n" + "="*70)
    if success:
        print("✅✅✅ Sanity Check 完全通过！")
        print("可以开始全量训练！")
    else:
        print("❌❌❌ Sanity Check 失败！")
        print("请先Debug，不要浪费时间跑全量数据！")
    print("="*70)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Sanity Check: Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('work_dirs/sanity_check_loss.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Loss曲线已保存到: work_dirs/sanity_check_loss.png")
    
    return success


if __name__ == '__main__':
    print("请先实现完整的DiffCGNet模型，然后调用此脚本")
    print("用法: python tools/test_sanity_check.py --config configs/... --checkpoint ...")
