import yaml
import torch
from pathlib import Path
from typing import Any
from datetime import datetime



class GPUMemoryLogger:
    """Logs GPU memory usage to file and checkpoints"""
    
    def __init__(self, log_dir: Path) -> None:
        self.log_dir: Path = log_dir
        self.log_file: Path = log_dir / "gpu_memory.yaml"
        self.per_epoch_stats: list[dict[str, Any]] = []
        
    
    def log_epoch_stats(
        self,
        epoch: int,
        allocated_gb: float,
        reserved_gb: float,
        max_allocated_gb: float,
    ) -> None:
        """Log GPU memory stats for an epoch"""
        stats: dict[str, Any] = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "allocated_gb": round(allocated_gb, 3),
            "reserved_gb": round(reserved_gb, 3),
            "max_allocated_gb": round(max_allocated_gb, 3),
        }
        
        self.per_epoch_stats.append(stats)
        
        with open(self.log_file, "w") as f:
            yaml.safe_dump(self.per_epoch_stats, f, indent = 2)    
    
    def get_current_stats(self) -> dict[str, float]:
        """Get current GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    
    
    def reset_max_memory(self) -> None:
        """Reset maximum memory tracking"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
