"""
Advanced multimodal fusion networks for medical AI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

class CrossModalAttentionFusion(nn.Module):
    """Advanced cross-modal attention fusion network"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dimension setup
        self.fusion_dim = config['fusion_dim']
        self.num_heads = config.get('num_heads', 8)
        
        # Cross-modal attention layers
        self.image_to_text_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        self.text_to_image_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Modality-specific projections
        self.image_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
        )
        
        # Output heads
        self.classification_head = nn.Linear(self.fusion_dim // 2, config['num_classes'])
        self.regression_head = nn.Linear(self.fusion_dim // 2, config.get('num_regression_targets', 5))
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(self.fusion_dim // 2, config['num_classes'])
        
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-modal attention"""
        
        # Ensure features have sequence dimension
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # Modality-specific projections
        image_proj = self.image_proj(image_features)
        text_proj = self.text_proj(text_features)
        
        # Cross-modal attention
        image_attended, image_attention_weights = self.image_to_text_attention(
            query=image_proj,
            key=text_proj,
            value=text_proj
        )
        
        text_attended, text_attention_weights = self.text_to_image_attention(
            query=text_proj,
            key=image_proj,
            value=image_proj
        )
        
        # Fusion
        image_fused = image_proj + image_attended
        text_fused = text_proj + text_attended
        
        # Concatenate and fuse
        combined = torch.cat([image_fused, text_fused], dim=-1)
        fused_features = self.fusion_layers(combined)
        
        # Global pooling
        if fused_features.dim() == 3:
            fused_features = fused_features.mean(dim=1)
        
        # Outputs
        classification_logits = self.classification_head(fused_features)
        regression_outputs = self.regression_head(fused_features)
        uncertainty_logits = self.uncertainty_head(fused_features)
        
        # Uncertainty estimation using softmax variance
        uncertainty = F.softmax(uncertainty_logits, dim=-1).var(dim=-1, keepdim=True)
        
        return {
            'classification': classification_logits,
            'regression': regression_outputs,
            'uncertainty': uncertainty,
            'attention_weights': {
                'image_to_text': image_attention_weights,
                'text_to_image': text_attention_weights
            },
            'embeddings': fused_features
        }

class HierarchicalFusionNetwork(nn.Module):
    """Hierarchical fusion network with multiple abstraction levels"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multiple fusion levels
        self.local_fusion = CrossModalAttentionFusion(config)
        
        # Global fusion with different dimensions
        global_config = config.copy()
        global_config['fusion_dim'] = config['fusion_dim'] * 2
        self.global_fusion = CrossModalAttentionFusion(global_config)
        
        # Hierarchical aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(config['fusion_dim'] // 2 + config['fusion_dim'], config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config['fusion_dim'], config['fusion_dim'] // 2)
        )
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hierarchical fusion forward pass"""
        
        # Local fusion (fine-grained)
        local_outputs = self.local_fusion(image_features, text_features)
        
        # Global fusion (coarse-grained)
        global_outputs = self.global_fusion(
            F.avg_pool1d(image_features.transpose(1, 2), kernel_size=2).transpose(1, 2),
            F.avg_pool1d(text_features.transpose(1, 2), kernel_size=2).transpose(1, 2)
        )
        
        # Aggregate local and global features
        aggregated_features = self.aggregation(
            torch.cat([local_outputs['embeddings'], global_outputs['embeddings']], dim=1)
        )
        
        # Final outputs
        classification_logits = self.local_fusion.classification_head(aggregated_features)
        regression_outputs = self.local_fusion.regression_head(aggregated_features)
        
        return {
            'classification': classification_logits,
            'regression': regression_outputs,
            'uncertainty': (local_outputs['uncertainty'] + global_outputs['uncertainty']) / 2,
            'attention_weights': {
                'local': local_outputs['attention_weights'],
                'global': global_outputs['attention_weights']
            },
            'embeddings': aggregated_features
        }

# Placeholder encoder classes to fix import errors
class MedicalImageEncoder(nn.Module):
    """Placeholder Medical Image Encoder"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, config['fusion_dim'])
        )
    
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(image)
        attention_weights = torch.softmax(features, dim=1)
        return features, attention_weights

class ClinicalTextEncoder(nn.Module):
    """Placeholder Clinical Text Encoder"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(10000, 512)
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config['fusion_dim'])
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, clinical_entities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        pooled = embeddings.mean(dim=1)
        features = self.encoder(pooled)
        attention_weights = torch.softmax(features, dim=1)
        return features, attention_weights

class ClinicalContextProcessor(nn.Module):
    """Process clinical context information"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Clinical factor embeddings
        self.factor_embeddings = nn.ModuleDict({
            'age': nn.Embedding(100, 32),  # 0-99 years
            'gender': nn.Embedding(3, 16),  # Male, Female, Unknown
            'history': nn.Linear(10, 64)   # Medical history factors
        })
        
        # Context fusion
        self.fusion = nn.Sequential(
            nn.Linear(32 + 16 + 64, config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
    
    def forward(self, clinical_context: Dict) -> torch.Tensor:
        """Process clinical context"""
        batch_size = next(iter(clinical_context.values())).size(0)
        
        # Embed different clinical factors
        age_emb = self.factor_embeddings['age'](clinical_context['age'].long())
        gender_emb = self.factor_embeddings['gender'](clinical_context['gender'].long())
        history_emb = self.factor_embeddings['history'](clinical_context['history'])
        
        # Concatenate and fuse
        context_features = torch.cat([age_emb, gender_emb, history_emb], dim=1)
        fused_context = self.fusion(context_features)
        
        return fused_context

class ClinicalFusionModel(nn.Module):
    """Complete clinical fusion model with encoders"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Encoders - using placeholder classes defined above
        self.image_encoder = MedicalImageEncoder(config)
        self.text_encoder = ClinicalTextEncoder(config)
        
        # Fusion network
        self.fusion_network = HierarchicalFusionNetwork(config)
        
        # Clinical context processor
        self.context_processor = ClinicalContextProcessor(config)
    
    def forward(self, image: torch.Tensor, text_data: Dict, clinical_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Complete forward pass with clinical context"""
        
        # Encode image
        image_features, image_attention = self.image_encoder(image)
        
        # Encode text
        text_features, text_attention = self.text_encoder(
            text_data['input_ids'],
            text_data['attention_mask'],
            text_data.get('clinical_entities')
        )
        
        # Process clinical context
        if clinical_context is not None:
            context_features = self.context_processor(clinical_context)
            # Incorporate context into features
            image_features = image_features + context_features
            text_features = text_features + context_features
        
        # Fusion
        fusion_outputs = self.fusion_network(
            image_features.unsqueeze(1),
            text_features.unsqueeze(1)
        )
        
        return {
            **fusion_outputs,
            'image_attention': image_attention,
            'text_attention': text_attention
        }