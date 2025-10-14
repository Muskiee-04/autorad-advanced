"""
Advanced medical image encoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import timm
import logging

class MedicalImageEncoder(nn.Module):
    """Advanced encoder for medical images"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Backbone selection
        backbone_name = config.get('image_backbone', 'densenet121')
        pretrained = config.get('pretrained', True)
        in_channels = config.get('in_channels', 1)
        
        try:
            if '3d' in backbone_name.lower():
                self.backbone = self._build_3d_backbone(backbone_name, in_channels)
            else:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    in_chans=in_channels,
                    num_classes=0,  # No classification head
                    features_only=False
                )
            
            # Get feature dimension
            if hasattr(self.backbone, 'num_features'):
                self.feature_dim = self.backbone.num_features
            else:
                # Estimate feature dimension
                self.feature_dim = self._get_feature_dim()
                
            # Feature projection
            self.feature_proj = nn.Linear(self.feature_dim, config['fusion_dim'])
            
            # Attention pooling
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=config['fusion_dim'],
                num_heads=config.get('num_heads', 8),
                batch_first=True
            )
            
        except Exception as e:
            self.logger.error(f"Error building image encoder: {e}")
            raise
    
    def _build_3d_backbone(self, backbone_name: str, in_channels: int) -> nn.Module:
        """Build 3D backbone for volumetric data"""
        # Placeholder for 3D backbones - would use MONAI in production
        class Simple3DCNN(nn.Module):
            def __init__(self, in_channels, feature_dim=512):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv3d(in_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1)
                )
                self.feature_dim = 128
                
            def forward(self, x):
                x = self.conv_layers(x)
                return x.view(x.size(0), -1)
        
        return Simple3DCNN(in_channels)
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension by forward pass"""
        dummy_input = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        return features.shape[-1]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention pooling"""
        # Extract features
        if x.dim() == 5:  # 3D data
            batch_size, channels, depth, height, width = x.shape
            x = x.view(batch_size, channels * depth, height, width)
        
        features = self.backbone(x)
        
        # Project to fusion dimension
        projected = self.feature_proj(features)
        
        # Apply attention pooling
        if projected.dim() == 2:
            projected = projected.unsqueeze(1)  # Add sequence dimension
        
        attended, attention_weights = self.attention_pool(
            projected, projected, projected
        )
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        
        return pooled, attention_weights

class MultiScaleImageEncoder(nn.Module):
    """Multi-scale image encoder for capturing different anatomical features"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multiple backbones for different scales
        self.global_encoder = MedicalImageEncoder(config)
        
        # Local feature encoder
        local_config = config.copy()
        local_config['image_backbone'] = 'resnet18'  # Lighter for local features
        self.local_encoder = MedicalImageEncoder(local_config)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(config['fusion_dim'] * 2, config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with multi-scale processing"""
        # Global features
        global_features, global_attn = self.global_encoder(x)
        
        # Local features (from center crop)
        batch_size, channels, height, width = x.shape
        crop_size = min(height, width) // 2
        start_h = (height - crop_size) // 2
        start_w = (width - crop_size) // 2
        local_patch = x[:, :, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        local_features, local_attn = self.local_encoder(local_patch)
        
        # Fusion
        fused_features = self.fusion(torch.cat([global_features, local_features], dim=1))
        
        attention_weights = {
            'global': global_attn,
            'local': local_attn
        }
        
        return fused_features, attention_weights