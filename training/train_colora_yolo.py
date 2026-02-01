"""
Complete CoLoRA training script for YOLOv9
Mainly an official training code adapted for CoLoRA layers

Run from project root:
    `python -m training.train_colora_yolo --config configs/colora_train.yaml`
"""

import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

# Add YOLOv9 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "models" / "official_yolov9"))

# YOLOv9 imports
from utils.general import (
    LOGGER,
    colorstr,
    increment_path,
    check_dataset,
    check_img_size,
    check_file,
    check_yaml,
    print_args,
    yaml_save,
    init_seeds,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
)
from utils.torch_utils import (
    select_device,
    torch_distributed_zero_first,
    ModelEMA,
    de_parallel,
    EarlyStopping,
)
from utils.dataloaders import create_dataloader
from utils.loss_tal import ComputeLoss
from utils.metrics import fitness
from utils.callbacks import Callbacks
from utils.loggers import Loggers
from models.experimental import attempt_load
from models.yolo import Model
import val as validate

# Adds project's root sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from models.conv_lora import CoLoRALayer
from utils.model_surgery import apply_colora_to_model, get_trainable_parameters
from utils.model_merge import merge_colora_model
from utils.gpu_logger import GPUMemoryLogger


class CoLoRATrainer:
    """Trainer that integrates CoLoRA with the original YOLOv9 training pipeline"""
    
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.device = select_device(config.get("device", ""))
        self.save_dir: Path = Path("")
        self.gpu_logger: Optional[GPUMemoryLogger] = None
        
        # Training parameters
        self.epochs: int = config.get("epochs", 100)
        self.batch_size: int = config.get("batch_size", 8)
        self.imgsz: int = config.get("img_size", 640)
        
        # CoLoRA parameters
        self.colora_rank: int = config.get("lora_rank", 4)
        self.colora_target_layers: list[str] = config.get("target_layers", ["backbone", "neck"])
        self.colora_lr: float = config.get("lora_lr", 1e-3)
        self.head_lr: float = config.get("head_lr", 1e-4)
        
        # Model and training state
        self.model: Optional[nn.Module] = None
        self.ema: Optional[ModelEMA] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.compute_loss: Optional[ComputeLoss] = None
        
        self.start_epoch: int = 0
        self.best_fitness: float = 0.0
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.dataset: Optional[Any] = None
        self.hyp: dict[str, Any] = {}
        
        # Training state variables
        self.maps: np.ndarray = np.array([])
        self.last_opt_step: int = -1
        self.nw: int = 0
        self.data_dict: Optional[dict[str, Any]] = None
        
        self.setup_logging()
    
    
    def setup_logging(self) -> None:
        """Setups experiment directory and logging"""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name: str = self.config.get("model", "yolov9-m").replace("/", "_")
        exp_name: str = f"{model_name}_colora_rank{self.colora_rank}"

        base_dir: Path = Path(self.config.get("project", "experiments/train-colora"))
        self.save_dir = increment_path(base_dir / exp_name / timestamp)
        self.save_dir.mkdir(parents = True, exist_ok = True)

        yaml_save(self.save_dir / "config.yaml", self.config)
        self.gpu_logger = GPUMemoryLogger(self.save_dir)

        LOGGER.info(f"{colorstr('CoLoRA Trainer:')} Save directory: {self.save_dir}")

    
    def setup_model(self) -> nn.Module:
        """Loads YOLOv9 model and apply CoLoRA"""
        LOGGER.info(f"{colorstr('CoLoRA:')} Loading model...")

        weights: str = self.config.get("weights", "yolov9-s.pt")
        cfg: str = self.config.get("cfg", "")
        data_config: str = self.config.get("data", "")
        
        hyp_path: str = self.config.get("hyp", "data/hyps/hyp.scratch-high.yaml")
        with open(hyp_path, errors = "ignore") as f:
            self.hyp = yaml.safe_load(f)
        
        with torch_distributed_zero_first(-1):
            self.data_dict = check_dataset(data_config)
        
        nc: int = self.data_dict["nc"]
        names: list[str] = self.data_dict["names"]
        
        pretrained: bool = weights.endswith(".pt")
        if pretrained:
            ckpt = torch.load(weights, map_location = "cpu", weights_only = False)
            model = Model(
                cfg or ckpt["model"].yaml,
                ch = 3,
                nc = nc,
                anchors = self.hyp.get("anchors"),
            ).to(self.device)
            
            csd = ckpt["model"].float().state_dict()
            csd = {k: v for k, v in csd.items() if model.state_dict()[k].shape == v.shape}
            model.load_state_dict(csd, strict = False)
            LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
        else:
            model = Model(
                cfg,
                ch = 3,
                nc = nc,
                anchors = self.hyp.get("anchors"),
            ).to(self.device)
        
        # Apply CoLoRA
        LOGGER.info(f"{colorstr('CoLoRA:')} Applying CoLoRA (rank = {self.colora_rank})...")
        num_converted, converted_names = apply_colora_to_model(
            model = model,
            target_layers = self.colora_target_layers,
            rank = self.colora_rank,
        )

        total_params, trainable_params, trainable_pct = get_trainable_parameters(model)
        
        LOGGER.info(f"{colorstr('CoLoRA:')} Converted {num_converted} layers")
        LOGGER.info(f"{colorstr('CoLoRA:')} Total params: {total_params:,}")
        LOGGER.info(f"{colorstr('CoLoRA:')} Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        
        with open(self.save_dir / "colora_info.txt", "w") as f:
            f.write(f"Converted layers: {num_converted}\n")
            f.write(f"Target layers: {self.colora_target_layers}\n")
            f.write(f"Rank: {self.colora_rank}\n")
            f.write(f"Total params: {total_params}\n")
            f.write(f"Trainable params: {trainable_params}\n")
            f.write(f"Trainable percentage: {trainable_pct:.2f}%\n")
            f.write("\nConverted layer names:\n")
            for name in converted_names:
                f.write(f"  - {name}\n")
        
        model.nc = nc
        model.names = names
        model.hyp = self.hyp
        
        self.model = model
        return model
    
    
    def setup_data(self) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setups data loaders"""
        data_config: str = self.config.get("data", "")

        with torch_distributed_zero_first(-1):
            self.data_dict = check_dataset(data_config)
        
        train_path: str = self.data_dict["train"]
        val_path: str = self.data_dict["val"]
        
        gs: int = max(int(self.model.stride.max()), 32)
        self.imgsz = check_img_size(self.imgsz, gs)
        
        self.train_loader, self.dataset = create_dataloader(
            path = train_path,
            imgsz = self.imgsz,
            batch_size = self.batch_size,
            stride = gs,
            single_cls = False,
            hyp = self.hyp,
            augment = True,
            cache = False,
            rect = False,
            rank = -1,
            workers = 8,
            image_weights = False,
            quad = False,
            prefix = colorstr("train: "),
        )
        
        self.val_loader = create_dataloader(
            path = val_path,
            imgsz = self.imgsz,
            batch_size = self.batch_size * 2,
            stride = gs,
            single_cls = False,
            hyp = self.hyp,
            cache = False,
            rect = True,
            rank = -1,
            workers = 8,
            pad = 0.5,
            prefix = colorstr("val: "),
        )[0]
        
        labels = np.concatenate(self.dataset.labels, 0)
        self.model.class_weights = labels_to_class_weights(labels, self.model.nc).to(self.device)
        
        return self.train_loader, self.val_loader
    
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for CoLoRA parameters"""
        lora_params: list[torch.Tensor] = []
        head_params: list[torch.Tensor] = []
        other_params: list[torch.Tensor] = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if "lora_pointwise" in name or "lora_layerwise" in name:
                lora_params.append(param)
            elif "detect" in name:
                head_params.append(param)
            else:
                other_params.append(param)
        
        # Parameter groups with different learning rates
        param_groups: list[dict[str, Any]] = []
        
        if lora_params:
            param_groups.append({"params": lora_params, "lr": self.colora_lr})
            LOGGER.info(f"{colorstr('Optimizer:')} LoRA params: {len(lora_params)}")
        
        if head_params:
            param_groups.append({"params": head_params, "lr": self.head_lr})
            LOGGER.info(f"{colorstr('Optimizer:')} Head params: {len(head_params)}")
        
        if other_params:
            param_groups.append({"params": other_params, "lr": self.hyp.get("lr0", 0.01)})
            LOGGER.info(f"{colorstr('Optimizer:')} Other params: {len(other_params)}")
        
        # Create optimizer
        optimizer_name: str = self.config.get("optimizer", "AdamW")
        if optimizer_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr = self.colora_lr,
                weight_decay = self.hyp.get("weight_decay", 0.0),
            )
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr = self.colora_lr,
                momentum = self.hyp.get("momentum", 0.937),
                weight_decay = self.hyp.get("weight_decay", 0.0005),
            )
        
        self.ema = ModelEMA(self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled = True)
        
        if self.config.get("cos_lr", False):
            lf = one_cycle(1, self.hyp["lrf"], self.epochs)
        else:
            def lf(x):
                return (1 - x / self.epochs) * (1.0 - self.hyp["lrf"]) + self.hyp["lrf"]
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lf)

        self.compute_loss = ComputeLoss(self.model)
        
        return self.optimizer
    
    
    def train_epoch(self, epoch: int) -> torch.Tensor:
        """Trains for one epoch"""
        self.model.train()
        
        if self.config.get("image_weights", False):
            cw = self.model.class_weights.cpu().numpy() * (1 - self.maps) ** 2 / self.model.nc
            iw = labels_to_image_weights(self.dataset.labels, nc = self.model.nc, class_weights = cw)
            self.dataset.indices = random.choices(range(self.dataset.n), weights = iw, k = self.dataset.n)
        
        mloss = torch.zeros(3, device = self.device)
        nb = len(self.train_loader)
        
        pbar = enumerate(self.train_loader)
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "cls_loss", "dfl_loss", "Instances", "Size"))
        pbar = tqdm(pbar, total = nb, bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}")
        
        self.optimizer.zero_grad()
        
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch  # number integrated batches
            
            # Warmup
            if ni <= self.nw:
                xi = [0, self.nw]
                accumulate = max(1, np.interp(ni, xi, [1, 64 / self.batch_size]).round())
                
                for j, x in enumerate(self.optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [0.0 if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [0.9, self.hyp["momentum"]])
            
            # Forward
            imgs = imgs.to(self.device, non_blocking = True).float() / 255
            
            with torch.cuda.amp.autocast(enabled = True):
                pred = self.model(imgs)
                loss, loss_items = self.compute_loss(pred, targets.to(self.device))
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Optimize
            if ni - self.last_opt_step >= accumulate:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.ema.update(self.model)
                self.last_opt_step = ni
            
            # Log
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = f"{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G"
            pbar.set_description(
                ("%11s" * 2 + "%11.4g" * 5) %
                (f"{epoch}/{self.epochs-1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
            )
        
        return mloss
    
    
    def validate(self, epoch: int) -> tuple[float, ...]:
        """Validates model"""
        results, _, _ = validate.run(
            data = self.data_dict,
            batch_size = self.batch_size * 2,
            imgsz = self.imgsz,
            half = True,
            model = self.ema.ema,
            single_cls = False,
            dataloader = self.val_loader,
            save_dir = self.save_dir,
            plots = False,
            compute_loss = self.compute_loss,
        )
        
        return results
    
    
    def save_checkpoint(
        self,
        epoch: int,
        fitness: float,
        results: tuple[float, ...],
        gpu_stats: dict[str, float],
    ) -> None:
        """Saves training checkpoint with GPU stats"""
        ckpt = {
            "epoch": epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "opt": self.config,
            "date": datetime.now().isoformat(),
            "gpu_stats": gpu_stats,
            "results": results,
        }
        
        weights_dir = self.save_dir / "weights"
        weights_dir.mkdir(exist_ok = True)
        
        torch.save(ckpt, weights_dir / "last.pt")
        if fitness == self.best_fitness:
            torch.save(ckpt, weights_dir / "best.pt")
        
        save_period = self.config.get("save_period", -1)
        if save_period > 0 and epoch % save_period == 0:
            torch.save(ckpt, weights_dir / f"epoch{epoch}.pt")
    
    
    def save_final_model(self) -> None:
        """Saves final model"""
        final_path = self.save_dir / "weights" / "final.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "hyp": self.hyp,
                "epoch": self.epochs,
                "best_fitness": self.best_fitness,
                "gpu_log": self.gpu_logger.per_epoch_stats if self.gpu_logger else [],
            },
            final_path,
        )
    
    
    def train(self) -> None:
        """Main training loop"""
        LOGGER.info(f"{colorstr('CoLoRA Trainer:')} Starting training for {self.epochs} epochs")
        
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        if self.gpu_logger:
            self.gpu_logger.reset_max_memory()
        
        self.maps = np.zeros(self.model.nc)
        self.last_opt_step = -1
        self.nw = max(round(self.hyp["warmup_epochs"] * len(self.train_loader)), 100)
        
        stopper = EarlyStopping(patience = self.config.get("patience", 100))
        callbacks = Callbacks()
        
        # Start training
        t0 = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            callbacks.run("on_train_epoch_start")
            
            if self.gpu_logger:
                self.gpu_logger.reset_max_memory()
            
            # Train one epoch
            mloss = self.train_epoch(epoch)
            self.scheduler.step()
            
            gpu_stats = {}
            if self.gpu_logger:
                gpu_stats = self.gpu_logger.get_current_stats()
                self.gpu_logger.log_epoch_stats(
                    epoch = epoch,
                    allocated_gb = gpu_stats.get("allocated_gb", 0.0),
                    reserved_gb = gpu_stats.get("reserved_gb", 0.0),
                    max_allocated_gb = gpu_stats.get("max_allocated_gb", 0.0),
                )
            
            # Validate
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                results = self.validate(epoch)
                
                fi = fitness(np.array(results).reshape(1, -1))
                if fi > self.best_fitness:
                    self.best_fitness = fi
                
                stop = stopper(epoch = epoch, fitness = fi)                
                self.save_checkpoint(epoch, fi, results, gpu_stats)
            
            LOGGER.info(
                f"{colorstr('GPU:')} Epoch {epoch}: "
                f"Allocated = {gpu_stats.get('allocated_gb', 0.0):.2f}GB, "
                f"Max = {gpu_stats.get('max_allocated_gb', 0.0):.2f}GB"
            )
            
            callbacks.run("on_train_epoch_end", epoch = epoch)
            
            if stop:
                LOGGER.info(f"{colorstr('EarlyStopping:')} Stopping training at epoch {epoch}")
                break
        
        # Training complete
        t1 = time.time()
        LOGGER.info(f"\n{self.epochs} epochs completed in {(t1 - t0) / 3600:.3f} hours")

        self.save_final_model()

        # Create merged model for inference
        merged_model = merge_colora_model(self.model, verbose = True)
        merged_path = self.save_dir / "weights" / "merged.pt"
        torch.save(merged_model.state_dict(), merged_path)
        
        LOGGER.info(f"{colorstr('CoLoRA Trainer:')} Training complete!")
        LOGGER.info(f"{colorstr('Models saved:')}")
        LOGGER.info(f"  - Training checkpoint: {self.save_dir / 'weights' / 'best.pt'}")
        LOGGER.info(f"  - Merged model: {merged_path}")
        LOGGER.info(f"  - GPU log: {self.save_dir / 'gpu_memory.json'}")


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data", type = str, required = True, help = "dataset.yaml path")
    parser.add_argument("--weights", type = str, default = "yolov9-s.pt", help = "initial weights path")
    parser.add_argument("--cfg", type = str, default = "", help = "model.yaml path")
    parser.add_argument("--hyp", type = str, default = "data/hyps/hyp.scratch-high.yaml", help = "hyperparameters path")
    
    # Training
    parser.add_argument("--epochs", type = int, default = 100)
    parser.add_argument("--batch-size", type = int, default = 8)
    parser.add_argument("--imgsz", "--img-size", type = int, default = 640)
    parser.add_argument("--device", default = "", help = "cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type = int, default = 8)
    parser.add_argument("--optimizer", type = str, choices = ["SGD", "AdamW"], default = "AdamW")
    
    # CoLoRA
    parser.add_argument("--lora-rank", type = int, default = 4)
    parser.add_argument("--lora-lr", type = float, default = 1e-3)
    parser.add_argument("--head-lr", type = float, default = 1e-4)
    parser.add_argument("--target-layers", nargs = "+", default = ["backbone", "neck"])
    
    # Other
    parser.add_argument("--project", default = "runs/train-colora", help = "save to project/name")
    parser.add_argument("--name", default = "exp", help = "experiment name")
    parser.add_argument("--exist-ok", action = "store_true", help = "existing project/name ok")
    parser.add_argument("--cos-lr", action = "store_true", help = "cosine LR scheduler")
    parser.add_argument("--patience", type = int, default = 100, help = "EarlyStopping patience")
    parser.add_argument("--save-period", type = int, default = -1, help = "save checkpoint every x epochs")
    parser.add_argument("--image-weights", action = "store_true", help = "use weighted image selection")
    
    return parser.parse_args()


def main() -> None:
    opt = parse_opt()
    
    # Create config dict
    config = vars(opt)
    
    # Override with config file if provided
    if hasattr(opt, "config") and opt.config:
        with open(opt.config, "r") as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # Initialize and run trainer
    trainer = CoLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
