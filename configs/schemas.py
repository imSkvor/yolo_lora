from dataclasses import dataclass, field

@dataclass
class CoLoRAConfig:
    """Configuration for CoLoRA fine-tuning"""
    enabled: bool = True
    rank: int = 4
    alpha: int = 8
    target_layers: list[str] = field(default_factory=lambda: ["backbone", "neck"])
    lr: float = 1e-3
    freeze_bn: bool = True

@dataclass
class TrainingConfig:
    data_path: str
    img_size: int = 640
    batch_size: int = 8
    
    epochs: int = 100
    warmup_epochs: int = 3

    colora: CoLoRAConfig = field(default_factory=CoLoRAConfig)

    def __post_init__(self) -> None:
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.img_size % 32 == 0, f"Image size {self.img_size} must be muliple of 32"
