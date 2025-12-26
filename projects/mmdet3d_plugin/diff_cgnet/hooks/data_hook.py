from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class InjectEpochHook(Hook):
    """
    直接设置模型的 current_epoch 属性
    """
    
    def __init__(self):
        self.current_epoch = 0
    
    def before_train_epoch(self, runner):
        """更新当前 epoch 并设置到模型"""
        self.current_epoch = runner.epoch
        
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        print(f"\n{'='*60}", flush=True)
        print(f"[InjectEpochHook] before_train_epoch called!", flush=True)
        print(f"[InjectEpochHook] runner.epoch = {runner.epoch}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # 获取模型（处理 DDP）
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        # 递归设置所有子模块的 current_epoch
        def set_epoch_recursive(module, epoch, depth=0):
            module.current_epoch = epoch
            # 只打印前几层
            if depth < 3:
                print(f"  {'  '*depth}Set {module.__class__.__name__}.current_epoch = {epoch}", flush=True)
            for child in module.children():
                set_epoch_recursive(child, epoch, depth+1)
        
        set_epoch_recursive(model, self.current_epoch)
        
        print(f"[InjectEpochHook] Finished setting current_epoch={self.current_epoch}\n", flush=True)
