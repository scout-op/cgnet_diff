"""
Anchor分析脚本
评估和统计anchor的分布信息
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze anchor distribution')
    parser.add_argument('--anchor-file', default='work_dirs/kmeans_anchors.pth',
                       help='Path to anchor file')
    parser.add_argument('--output-dir', default='work_dirs/anchor_analysis',
                       help='Output directory for analysis results')
    return parser.parse_args()


def bezier_interpolate(ctrl_points, num_points=50):
    """贝塞尔插值"""
    from math import factorial
    
    def comb(n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    n_control = len(ctrl_points)
    degree = n_control - 1
    
    t = np.linspace(0, 1, num_points)
    points = np.zeros((num_points, 2))
    
    for i in range(n_control):
        coef = comb(degree, i) * np.power(1 - t, degree - i) * np.power(t, i)
        points += coef[:, np.newaxis] * ctrl_points[i]
    
    return points


def compute_curve_direction(ctrl_points):
    """计算曲线的主方向"""
    start = ctrl_points[0]
    end = ctrl_points[-1]
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    
    angle = np.arctan2(dy, dx) * 180 / np.pi
    
    return angle


def compute_curve_length(ctrl_points, num_samples=50):
    """计算曲线长度"""
    points = bezier_interpolate(ctrl_points, num_samples)
    
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    total_length = lengths.sum()
    
    return total_length


def compute_curvature(ctrl_points):
    """计算曲线的平均曲率"""
    points = bezier_interpolate(ctrl_points, 50)
    
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)
    
    return curvature.mean()


def classify_anchor_type(ctrl_points):
    """分类anchor类型"""
    angle = compute_curve_direction(ctrl_points)
    curvature = compute_curvature(ctrl_points)
    
    if curvature < 0.01:
        return 'straight'
    elif angle > 45:
        return 'left_turn'
    elif angle < -45:
        return 'right_turn'
    elif abs(angle) < 45:
        if curvature < 0.05:
            return 'straight'
        else:
            return 'curved'
    else:
        return 'other'


def analyze_anchors(anchor_file, output_dir):
    """分析anchor分布"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("🔍 Anchor分布分析")
    print("="*70)
    
    data = torch.load(anchor_file)
    anchors = data['anchors'].numpy()
    
    print(f"\n基本信息:")
    print(f"  Anchor数量: {len(anchors)}")
    print(f"  控制点数: {anchors.shape[1]}")
    print(f"  坐标维度: {anchors.shape[2]}")
    print(f"  总形状: {anchors.shape}")
    
    print(f"\n坐标范围:")
    print(f"  X: [{anchors[:, :, 0].min():.2f}, {anchors[:, :, 0].max():.2f}]")
    print(f"  Y: [{anchors[:, :, 1].min():.2f}, {anchors[:, :, 1].max():.2f}]")
    
    print("\n" + "-"*70)
    print("统计分析:")
    print("-"*70)
    
    lengths = []
    curvatures = []
    directions = []
    types = {'straight': 0, 'left_turn': 0, 'right_turn': 0, 'curved': 0, 'other': 0}
    
    for i, anchor in enumerate(anchors):
        length = compute_curve_length(anchor)
        curvature = compute_curvature(anchor)
        direction = compute_curve_direction(anchor)
        anchor_type = classify_anchor_type(anchor)
        
        lengths.append(length)
        curvatures.append(curvature)
        directions.append(direction)
        types[anchor_type] += 1
    
    lengths = np.array(lengths)
    curvatures = np.array(curvatures)
    directions = np.array(directions)
    
    print(f"\n长度统计:")
    print(f"  平均: {lengths.mean():.2f} m")
    print(f"  最小: {lengths.min():.2f} m")
    print(f"  最大: {lengths.max():.2f} m")
    print(f"  标准差: {lengths.std():.2f} m")
    
    print(f"\n曲率统计:")
    print(f"  平均: {curvatures.mean():.4f}")
    print(f"  最小: {curvatures.min():.4f}")
    print(f"  最大: {curvatures.max():.4f}")
    
    print(f"\n方向统计:")
    print(f"  平均角度: {directions.mean():.2f}°")
    print(f"  角度范围: [{directions.min():.2f}°, {directions.max():.2f}°]")
    
    print(f"\n类型分布:")
    total = sum(types.values())
    for anchor_type, count in sorted(types.items(), key=lambda x: -x[1]):
        percentage = count / total * 100
        print(f"  {anchor_type:12s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "-"*70)
    print("多样性分析:")
    print("-"*70)
    
    anchors_flat = anchors.reshape(len(anchors), -1)
    distances = pdist(anchors_flat, metric='euclidean')
    
    print(f"\nAnchor间距离:")
    print(f"  平均: {distances.mean():.2f}")
    print(f"  最小: {distances.min():.2f} (最相似的两个anchor)")
    print(f"  最大: {distances.max():.2f} (最不同的两个anchor)")
    print(f"  标准差: {distances.std():.2f}")
    
    diversity_score = distances.mean() / distances.std()
    print(f"\n多样性得分: {diversity_score:.2f}")
    if diversity_score > 1.0:
        print("  ✅ 多样性良好")
    else:
        print("  ⚠️ 多样性较低，anchor可能过于相似")
    
    print("\n" + "-"*70)
    print("可视化生成:")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    ax = axes[0, 0]
    ax.hist(lengths, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Count')
    ax.set_title('Anchor Length Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.hist(curvatures, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Curvature')
    ax.set_ylabel('Count')
    ax.set_title('Anchor Curvature Distribution')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.hist(directions, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Direction (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('Anchor Direction Distribution')
    ax.axvline(0, color='r', linestyle='--', label='Forward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    type_names = list(types.keys())
    type_counts = [types[t] for t in type_names]
    ax.bar(type_names, type_counts, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Anchor Type Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax = axes[1, 1]
    dist_matrix = squareform(distances)
    im = ax.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Anchor Index')
    ax.set_ylabel('Anchor Index')
    ax.set_title('Pairwise Distance Matrix')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    for i, anchor in enumerate(anchors[::5]):
        points = bezier_interpolate(anchor)
        ax.plot(points[:, 0], points[:, 1], alpha=0.5, linewidth=1)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Sample Anchors (every 5th)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path = f'{output_dir}/anchor_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 统计图已保存: {save_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("📊 分析报告")
    print("="*70)
    
    report = []
    report.append("Anchor质量评估:\n")
    
    if len(anchors) >= 30:
        report.append("✅ 数量充足 (>= 30)")
    else:
        report.append("⚠️ 数量较少 (< 30)")
    
    if types['straight'] > len(anchors) * 0.3:
        report.append("✅ 直行车道充足")
    else:
        report.append("⚠️ 直行车道较少")
    
    if types['left_turn'] > 5 and types['right_turn'] > 5:
        report.append("✅ 转弯车道覆盖良好")
    else:
        report.append("⚠️ 转弯车道覆盖不足")
    
    if diversity_score > 1.0:
        report.append("✅ 多样性良好")
    else:
        report.append("⚠️ 多样性不足")
    
    if lengths.std() / lengths.mean() > 0.3:
        report.append("✅ 长度分布多样")
    else:
        report.append("⚠️ 长度分布单一")
    
    print("\n".join(report))
    
    with open(f'{output_dir}/analysis_report.txt', 'w') as f:
        f.write("Anchor分布分析报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"基本信息:\n")
        f.write(f"  数量: {len(anchors)}\n")
        f.write(f"  形状: {anchors.shape}\n\n")
        f.write(f"坐标范围:\n")
        f.write(f"  X: [{anchors[:, :, 0].min():.2f}, {anchors[:, :, 0].max():.2f}]\n")
        f.write(f"  Y: [{anchors[:, :, 1].min():.2f}, {anchors[:, :, 1].max():.2f}]\n\n")
        f.write(f"长度统计:\n")
        f.write(f"  平均: {lengths.mean():.2f} m\n")
        f.write(f"  范围: [{lengths.min():.2f}, {lengths.max():.2f}] m\n")
        f.write(f"  标准差: {lengths.std():.2f} m\n\n")
        f.write(f"类型分布:\n")
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            f.write(f"  {t}: {c} ({c/total*100:.1f}%)\n")
        f.write(f"\n多样性得分: {diversity_score:.2f}\n\n")
        f.write("质量评估:\n")
        f.write("\n".join(report))
    
    print(f"\n✅ 分析报告已保存: {output_dir}/analysis_report.txt")
    
    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    
    return {
        'num_anchors': len(anchors),
        'length_mean': lengths.mean(),
        'length_std': lengths.std(),
        'diversity_score': diversity_score,
        'types': types
    }


if __name__ == '__main__':
    args = parse_args()
    
    try:
        stats = analyze_anchors(args.anchor_file, args.output_dir)
        
        print("\n" + "="*70)
        print("📋 快速总结")
        print("="*70)
        print(f"Anchor数量: {stats['num_anchors']}")
        print(f"平均长度: {stats['length_mean']:.2f} m")
        print(f"多样性得分: {stats['diversity_score']:.2f}")
        print(f"主要类型: {max(stats['types'].items(), key=lambda x: x[1])}")
        
        if stats['diversity_score'] > 1.0 and stats['num_anchors'] >= 30:
            print("\n✅ Anchor质量良好，可以用于训练！")
        else:
            print("\n⚠️ Anchor质量一般，建议检查数据或调整聚类参数")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
