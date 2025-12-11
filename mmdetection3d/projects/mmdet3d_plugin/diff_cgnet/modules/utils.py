import torch
import numpy as np
from math import factorial


def comb(n, k):
    """组合数 C(n,k)"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def fit_bezier(points, n_control=4):
    """
    将点序列拟合为贝塞尔曲线控制点
    
    Args:
        points: np.ndarray or LineString, shape (n_points, 2)
        n_control: int, 控制点数量（默认4，即三次贝塞尔）
    
    Returns:
        control_points: np.ndarray, shape (n_control, 2)
    """
    # Convert LineString to numpy array if needed
    if hasattr(points, 'coords'):
        # It's a shapely LineString
        points = np.array(points.coords)
    elif not isinstance(points, np.ndarray):
        points = np.array(points)
    
    if len(points) < 10:
        points = np.linspace(points[0], points[-1], num=10)
    
    n_points = len(points)
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)
    
    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * \
                      np.power(1 - t[i], n_control - 1 - j) * \
                      np.power(t[i], j)
    
    A_BE = A[:, 1:-1]
    points_BE = points - np.stack([
        (A[:, 0] * points[0][0] + A[:, -1] * points[-1][0]),
        (A[:, 0] * points[0][1] + A[:, -1] * points[-1][1])
    ]).T
    
    try:
        conts = np.linalg.lstsq(A_BE, points_BE, rcond=None)
    except:
        raise Exception("Bezier fitting failed! Check if points are valid.")
    
    res = conts[0]
    fin_res = np.r_[[points[0]], res, [points[-1]]]
    
    return fin_res


def bezier_interpolate(ctrl_points, num_points=20):
    """
    贝塞尔曲线插值
    
    Args:
        ctrl_points: torch.Tensor, shape (..., 4, 2) 或 (..., n_control, 2)
        num_points: int, 插值点数量
    
    Returns:
        points: torch.Tensor, shape (..., num_points, 2)
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    
    n_control = ctrl_points.shape[-2]
    degree = n_control - 1
    
    t = torch.linspace(0, 1, num_points, device=device, dtype=dtype)
    
    shape = ctrl_points.shape[:-2]
    t = t.view(*([1] * len(shape)), num_points, 1)
    
    points = torch.zeros(*shape, num_points, 2, device=device, dtype=dtype)
    
    for i in range(n_control):
        coef = comb(degree, i) * torch.pow(1 - t, degree - i) * torch.pow(t, i)
        ctrl_i = ctrl_points[..., i:i+1, :]
        points = points + coef * ctrl_i
    
    return points


def cubic_bezier_interpolate(ctrl_points, num_points=20):
    """
    三次贝塞尔曲线插值（优化版本，专门针对4个控制点）
    
    Args:
        ctrl_points: torch.Tensor, shape (..., 4, 2)
        num_points: int, 插值点数量
    
    Returns:
        points: torch.Tensor, shape (..., num_points, 2)
    """
    device = ctrl_points.device
    dtype = ctrl_points.dtype
    
    t = torch.linspace(0, 1, num_points, device=device, dtype=dtype)
    
    shape = ctrl_points.shape[:-2]
    t = t.view(*([1] * len(shape)), num_points, 1)
    
    P0 = ctrl_points[..., 0:1, :]
    P1 = ctrl_points[..., 1:2, :]
    P2 = ctrl_points[..., 2:3, :]
    P3 = ctrl_points[..., 3:4, :]
    
    coef0 = (1 - t) ** 3
    coef1 = 3 * (1 - t) ** 2 * t
    coef2 = 3 * (1 - t) * t ** 2
    coef3 = t ** 3
    
    points = coef0 * P0 + coef1 * P1 + coef2 * P2 + coef3 * P3
    
    return points


def normalize_coords(coords, pc_range):
    """
    将坐标从Lidar坐标系归一化到[0, 1]
    
    Args:
        coords: torch.Tensor, shape (..., 2), (x, y)
        pc_range: list, [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        normalized_coords: torch.Tensor, shape (..., 2)
    """
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    normalized = coords.clone()
    normalized[..., 0] = (coords[..., 0] - x_min) / (x_max - x_min)
    normalized[..., 1] = (coords[..., 1] - y_min) / (y_max - y_min)
    
    normalized = torch.clamp(normalized, 0.01, 0.99)
    
    return normalized


def denormalize_coords(normalized_coords, pc_range):
    """
    将归一化坐标转回Lidar坐标系
    
    Args:
        normalized_coords: torch.Tensor, shape (..., 2)
        pc_range: list, [x_min, y_min, z_min, x_max, y_max, z_max]
    
    Returns:
        coords: torch.Tensor, shape (..., 2)
    """
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    coords = normalized_coords.clone()
    coords[..., 0] = normalized_coords[..., 0] * (x_max - x_min) + x_min
    coords[..., 1] = normalized_coords[..., 1] * (y_max - y_min) + y_min
    
    return coords


def chamfer_distance(pred_points, gt_points, eps=1e-6):
    """
    计算Chamfer Distance
    
    Args:
        pred_points: torch.Tensor, shape (N, K, 2)
        gt_points: torch.Tensor, shape (M, K, 2)
        eps: float, 防止除零
    
    Returns:
        distance: torch.Tensor, scalar
    """
    pred_points = pred_points.unsqueeze(1)
    gt_points = gt_points.unsqueeze(0)
    
    dist_matrix = torch.norm(pred_points - gt_points, dim=-1, p=2)
    
    dist_pred_to_gt = dist_matrix.min(dim=1)[0].mean()
    dist_gt_to_pred = dist_matrix.min(dim=0)[0].mean()
    
    return (dist_pred_to_gt + dist_gt_to_pred) / 2
