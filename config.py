"""
Configuration management for AutoRad
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_dir: str = "./data"
    image_size: List[int] = (224, 224)
    hu_normalization: bool = True
    hu_min: int = -1000
    hu_max: int = 2000
    augment_3d: bool = False
    text_max_length: int = 512
    batch_size: int = 8
    num_workers: int = 2

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    image_backbone: str = "densenet121"
    text_backbone: str = "emilyalsentzer/Bio_ClinicalBERT"
    fusion_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    num_classes: int = 15
    use_3d: bool = False
    llm_model: str = "microsoft/BioGPT-Large"
    pretrained: bool = True
    in_channels: int = 1

@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    cls_weight: float = 1.0
    reg_weight: float = 0.5
    clinical_weight: float = 0.1

@dataclass
class AutoRadConfig:
    """Complete AutoRad configuration"""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    deployment: Dict = None
    
    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            deployment=config_dict.get('deployment', {})
        )