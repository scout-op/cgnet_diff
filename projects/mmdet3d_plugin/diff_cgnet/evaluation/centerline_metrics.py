import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import networkx as nx


def chamfer_distance_metric(pred_lines, gt_lines, threshold=1.0):
    """
    计算Chamfer Distance指标
    
    Args:
        pred_lines: List[np.ndarray], 预测的中心线
        gt_lines: List[np.ndarray], GT中心线
        threshold: float, 匹配阈值
    
    Returns:
        chamfer_dist: float
    """
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return float('inf')
    
    total_dist = 0
    for pred_line in pred_lines:
        min_dist = float('inf')
        for gt_line in gt_lines:
            dist = compute_chamfer(pred_line, gt_line)
            min_dist = min(min_dist, dist)
        total_dist += min_dist
    
    return total_dist / len(pred_lines)


def compute_chamfer(line1, line2):
    """计算两条线的Chamfer距离"""
    dist_matrix = cdist(line1, line2, metric='euclidean')
    
    dist_1_to_2 = dist_matrix.min(axis=1).mean()
    dist_2_to_1 = dist_matrix.min(axis=0).mean()
    
    return (dist_1_to_2 + dist_2_to_1) / 2


def compute_geo_f1(pred_lines, gt_lines, threshold=1.0):
    """
    计算GEO F1分数
    
    Args:
        pred_lines: List[np.ndarray]
        gt_lines: List[np.ndarray]
        threshold: float, 匹配阈值（米）
    
    Returns:
        f1: float
        precision: float
        recall: float
    """
    if len(pred_lines) == 0:
        return 0.0, 0.0, 0.0
    if len(gt_lines) == 0:
        return 0.0, 0.0, 0.0
    
    pred_points = np.vstack(pred_lines)
    gt_points = np.vstack(gt_lines)
    
    dist_matrix = cdist(pred_points, gt_points, metric='euclidean')
    
    pred_matched = (dist_matrix.min(axis=1) < threshold).sum()
    gt_matched = (dist_matrix.min(axis=0) < threshold).sum()
    
    precision = pred_matched / len(pred_points) if len(pred_points) > 0 else 0
    recall = gt_matched / len(gt_points) if len(gt_points) > 0 else 0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return f1, precision, recall


def compute_topo_f1(pred_lines, pred_topology, gt_lines, gt_topology, threshold=1.0):
    """
    计算TOPO F1分数（考虑拓扑连接）
    
    Args:
        pred_lines: List[np.ndarray]
        pred_topology: np.ndarray, [N, N]
        gt_lines: List[np.ndarray]
        gt_topology: np.ndarray, [M, M]
        threshold: float
    
    Returns:
        topo_f1: float
    """
    if len(pred_lines) == 0 or len(gt_lines) == 0:
        return 0.0
    
    pred_graph = build_graph(pred_lines, pred_topology)
    gt_graph = build_graph(gt_lines, gt_topology)
    
    matched_edges = 0
    total_pred_edges = pred_graph.number_of_edges()
    total_gt_edges = gt_graph.number_of_edges()
    
    for u, v in pred_graph.edges():
        for gu, gv in gt_graph.edges():
            if is_edge_matched(
                pred_lines[u], pred_lines[v],
                gt_lines[gu], gt_lines[gv],
                threshold
            ):
                matched_edges += 1
                break
    
    precision = matched_edges / total_pred_edges if total_pred_edges > 0 else 0
    recall = matched_edges / total_gt_edges if total_gt_edges > 0 else 0
    
    if precision + recall > 0:
        topo_f1 = 2 * precision * recall / (precision + recall)
    else:
        topo_f1 = 0.0
    
    return topo_f1


def build_graph(lines, topology):
    """构建NetworkX图"""
    G = nx.DiGraph()
    
    for i in range(len(lines)):
        G.add_node(i)
    
    for i in range(len(topology)):
        for j in range(len(topology)):
            if topology[i, j] > 0.5:
                G.add_edge(i, j)
    
    return G


def is_edge_matched(line1_start, line1_end, line2_start, line2_end, threshold):
    """判断两条边是否匹配"""
    dist_start = np.linalg.norm(line1_start[-1] - line2_start[0])
    dist_end = np.linalg.norm(line1_end[-1] - line2_end[0])
    
    return dist_start < threshold and dist_end < threshold


def compute_apls(pred_lines, pred_topology, gt_lines, gt_topology):
    """
    计算APLS（Average Path Length Similarity）
    """
    pred_graph = build_graph(pred_lines, pred_topology)
    gt_graph = build_graph(gt_lines, gt_topology)
    
    try:
        pred_lengths = dict(nx.all_pairs_dijkstra_path_length(pred_graph))
        gt_lengths = dict(nx.all_pairs_dijkstra_path_length(gt_graph))
    except:
        return 0.0
    
    total_diff = 0
    count = 0
    
    for i in pred_lengths:
        for j in pred_lengths[i]:
            if i in gt_lengths and j in gt_lengths[i]:
                pred_len = pred_lengths[i][j]
                gt_len = gt_lengths[i][j]
                total_diff += abs(pred_len - gt_len) / max(pred_len, gt_len, 1e-6)
                count += 1
    
    if count == 0:
        return 0.0
    
    apls = 1.0 - (total_diff / count)
    return max(0.0, apls)


def evaluate_all_metrics(pred_results, gt_data, thresholds=[0.5, 1.0, 1.5]):
    """
    计算所有评估指标
    
    Args:
        pred_results: List[dict], 预测结果
        gt_data: List[dict], GT数据
        thresholds: List[float], 评估阈值
    
    Returns:
        metrics: dict, 所有指标
    """
    metrics = {
        'geo_f1': [],
        'topo_f1': [],
        'apls': [],
        'chamfer': []
    }
    
    for pred, gt in zip(pred_results, gt_data):
        pred_lines = pred['centerlines']
        gt_lines = gt['centerlines']
        
        pred_topo = pred.get('topology', None)
        gt_topo = gt.get('topology', None)
        
        for threshold in thresholds:
            geo_f1, _, _ = compute_geo_f1(pred_lines, gt_lines, threshold)
            metrics['geo_f1'].append(geo_f1)
            
            chamfer = chamfer_distance_metric(pred_lines, gt_lines, threshold)
            metrics['chamfer'].append(chamfer)
            
            if pred_topo is not None and gt_topo is not None:
                topo_f1 = compute_topo_f1(
                    pred_lines, pred_topo,
                    gt_lines, gt_topo,
                    threshold
                )
                metrics['topo_f1'].append(topo_f1)
                
                apls = compute_apls(
                    pred_lines, pred_topo,
                    gt_lines, gt_topo
                )
                metrics['apls'].append(apls)
    
    result = {}
    for key in metrics:
        if len(metrics[key]) > 0:
            result[key] = np.mean(metrics[key])
        else:
            result[key] = 0.0
    
    return result
