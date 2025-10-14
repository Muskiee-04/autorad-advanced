"""
Advanced medical text encoders
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple, List
import logging

class ClinicalTextEncoder(nn.Module):
    """Advanced encoder for medical text with clinical knowledge"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Clinical language model
        model_name = config.get('text_backbone', 'emilyalsentzer/Bio_ClinicalBERT')
        
        try:
            self.language_model = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.language_model.config.hidden_size
            
            # Clinical concept projection
            self.concept_proj = nn.Linear(self.hidden_size, config['fusion_dim'])
            
            # Attention mechanism for important phrases
            self.phrase_attention = nn.MultiheadAttention(
                embed_dim=config['fusion_dim'],
                num_heads=config.get('num_heads', 8),
                batch_first=True
            )
            
            # Clinical entity enhancement
            self.entity_enhancer = ClinicalEntityEnhancer(config)
            
        except Exception as e:
            self.logger.error(f"Error building text encoder: {e}")
            raise
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                clinical_entities: Optional[List] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with clinical entity enhancement"""
        
        # Get language model outputs
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        
        # Last hidden states
        hidden_states = outputs.last_hidden_state
        
        # Project to fusion dimension
        projected = self.concept_proj(hidden_states)
        
        # Apply clinical entity enhancement
        if clinical_entities:
            enhanced_states = self.entity_enhancer(projected, clinical_entities, attention_mask)
        else:
            enhanced_states = projected
        
        # Attention pooling over sequence
        attended, attention_weights = self.phrase_attention(
            enhanced_states, enhanced_states, enhanced_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        
        return pooled, attention_weights

class ClinicalEntityEnhancer(nn.Module):
    """Enhance clinical entity representations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Entity type embeddings
        self.entity_embeddings = nn.Embedding(50, config['fusion_dim'])  # 50 entity types
        
        # Enhancement layers
        self.enhancement = nn.Sequential(
            nn.Linear(config['fusion_dim'] * 2, config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                clinical_entities: List,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Enhance hidden states with clinical entity information"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        enhanced_states = hidden_states.clone()
        
        for batch_idx, entities in enumerate(clinical_entities):
            for entity in entities:
                start_idx = entity.get('start', 0)
                end_idx = entity.get('end', seq_len - 1)
                entity_type = entity.get('label', 'UNK')
                
                if start_idx < seq_len and end_idx < seq_len:
                    # Get entity type embedding
                    entity_type_id = self._get_entity_type_id(entity_type)
                    entity_emb = self.entity_embeddings(entity_type_id)
                    
                    # Enhance entity span
                    entity_span = enhanced_states[batch_idx, start_idx:end_idx+1]
                    enhanced_span = self.enhancement(
                        torch.cat([entity_span, entity_emb.expand(entity_span.size(0), -1)], dim=-1)
                    )
                    
                    enhanced_states[batch_idx, start_idx:end_idx+1] = enhanced_span
        
        return enhanced_states
    
    def _get_entity_type_id(self, entity_type: str) -> torch.Tensor:
        """Convert entity type to ID"""
        # Simple hash-based mapping
        entity_hash = hash(entity_type) % 50
        return torch.tensor(entity_hash, dtype=torch.long)

class MultiModalTextEncoder(nn.Module):
    """Multi-modal text encoder that combines different representations"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Main clinical encoder
        self.clinical_encoder = ClinicalTextEncoder(config)
        
        # Measurement encoder
        self.measurement_encoder = nn.Sequential(
            nn.Linear(10, 64),  # 10 measurement features
            nn.ReLU(),
            nn.Linear(64, config['fusion_dim'] // 4)
        )
        
        # Temporal encoder for prior reports
        self.temporal_encoder = nn.LSTM(
            input_size=config['fusion_dim'],
            hidden_size=config['fusion_dim'] // 2,
            num_layers=2,
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(config['fusion_dim'] + config['fusion_dim'] // 4 + config['fusion_dim'] // 2, 
                     config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
    
    def forward(self, current_input: Dict, prior_input: Optional[Dict] = None) -> torch.Tensor:
        """Encode current and prior text with temporal context"""
        
        # Encode current report
        current_emb, _ = self.clinical_encoder(
            current_input['input_ids'],
            current_input['attention_mask'],
            current_input.get('clinical_entities')
        )
        
        # Encode measurements
        measurements = current_input.get('measurements', torch.zeros(current_emb.size(0), 10))
        measurement_emb = self.measurement_encoder(measurements)
        
        # Encode temporal context if prior available
        if prior_input is not None:
            prior_emb, _ = self.clinical_encoder(
                prior_input['input_ids'],
                prior_input['attention_mask'],
                prior_input.get('clinical_entities')
            )
            
            # Temporal encoding
            temporal_seq = torch.stack([prior_emb, current_emb], dim=1)
            temporal_emb, _ = self.temporal_encoder(temporal_seq)
            temporal_emb = temporal_emb[:, -1, :]  # Last timestep
        else:
            temporal_emb = torch.zeros(current_emb.size(0), self.config['fusion_dim'] // 2)
        
        # Fuse all representations
        fused_emb = self.fusion(
            torch.cat([current_emb, measurement_emb, temporal_emb], dim=1)
        )
        
        return fused_emb