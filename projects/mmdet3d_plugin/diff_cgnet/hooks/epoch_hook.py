from mmcv.runner import HOOKS, Hook
import torch.distributed as dist
import os

# 使用环境变量存储 epoch（进程级别共享）
ENV_EPOCH_KEY = 'DIFF_CGNET_CURRENT_EPOCH'


def get_global_epoch():
    """获取当前全局 epoch"""
    return int(os.environ.get(ENV_EPOCH_KEY, '0'))


@HOOKS.register_module()
class EpochUpdateHook(Hook):
    """
    在每个 epoch 开始时更新环境变量中的 epoch
    """
    
    def __init__(self):
        pass
    
    def before_train_epoch(self, runner):
        """
        在每个训练 epoch 开始前调用（所有进程都会执行）
        """
        epoch = runner.epoch  # 0-indexed
        os.environ[ENV_EPOCH_KEY] = str(epoch)
        
        # 打印日志
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"\n[EpochUpdateHook] Set ENV epoch={epoch}")
