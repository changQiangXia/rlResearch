"""
通用日志系统，支持控制台输出和 TensorBoard 写入
"""
import os
from typing import Dict, Any


class Logger:
    """基础日志记录器"""
    
    def __init__(self, log_dir: str = None, use_tensorboard: bool = False):
        """
        Args:
            log_dir: 日志保存目录
            use_tensorboard: 是否使用 TensorBoard
        """
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if use_tensorboard and log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir)
            except ImportError:
                print("Warning: tensorboard not available")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_dict(self, data: Dict[str, float], step: int):
        """批量记录字典数据"""
        for tag, value in data.items():
            self.log_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """记录直方图（用于 Q 值分布等）"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def print_console(self, step: int, metrics: Dict[str, Any]):
        """打印到控制台"""
        msg = f"Step {step:6d} | "
        msg += " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                          for k, v in metrics.items()])
        print(msg)
    
    def close(self):
        """关闭日志记录器"""
        if self.writer:
            self.writer.close()
