"""
Advanced multimodal training with medical-specific loss functions
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class ClinicalConsistencyLoss(nn.Module):
    """Loss enforcing clinical consistency in predictions"""
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions: torch.Tensor, clinical_rules: Dict) -> torch.Tensor:
        # Implement clinical rule-based consistency checks
        loss = torch.tensor(0.0, device=predictions.device)
        
        # Example rule: Certain findings shouldn't co-occur
        # This would be expanded with real clinical rules
        if 'contradictory_findings' in clinical_rules:
            for finding_pair in clinical_rules['contradictory_findings']:
                i, j = finding_pair
                # Penalize high probabilities for contradictory findings
                contradiction_loss = torch.min(predictions[:, i], predictions[:, j])
                loss += contradiction_loss.mean()
        
        return loss

class MedicalMultimodalTrainer:
    """Advanced trainer for medical multimodal models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Medical-specific loss functions
        self.criterion = {
            'classification': nn.BCEWithLogitsLoss(),
            'regression': nn.MSELoss(),
            'contrastive': nn.TripletMarginLoss(),
            'clinical': ClinicalConsistencyLoss()
        }
    
    def train_epoch(self, 
                   model: nn.Module, 
                   dataloader: DataLoader, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[Any] = None) -> Dict[str, float]:
        """Train for one epoch with medical-specific enhancements"""
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass
            outputs = model(batch['image'], batch)
            
            # Medical-specific multi-task loss
            loss = self._compute_medical_loss(outputs, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': optimizer.param_groups[0]['lr']
            })
        
        return {'train_loss': total_loss / len(dataloader)}
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """Prepare batch for training"""
        # Move tensors to device
        prepared_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device)
            else:
                prepared_batch[key] = value
        return prepared_batch
    
    def _compute_medical_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """Compute comprehensive medical loss"""
        total_loss = 0
        
        # Classification loss
        if 'classification' in outputs and 'labels' in batch:
            cls_loss = self.criterion['classification'](
                outputs['classification'], 
                batch['labels']['binary']
            )
            total_loss += cls_loss * self.config.get('cls_weight', 1.0)
        
        # Regression loss for measurements
        if 'regression' in outputs and 'labels' in batch:
            reg_loss = self.criterion['regression'](
                outputs['regression'],
                batch['labels']['measurements']
            )
            total_loss += reg_loss * self.config.get('reg_weight', 0.5)
        
        # Clinical consistency loss
        clinical_rules = batch.get('clinical_rules', {})
        clinical_loss = self.criterion['clinical'](
            torch.sigmoid(outputs.get('classification', torch.zeros(1, device=self.device))), 
            clinical_rules
        )
        total_loss += clinical_loss * self.config.get('clinical_weight', 0.1)
        
        return total_loss
    
    def validate(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Comprehensive medical validation"""
        model.eval()
        val_metrics = {
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'val_auc': 0.0,
            'val_f1': 0.0
        }
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                batch = self._prepare_batch(batch)
                outputs = model(batch['image'], batch)
                
                # Calculate loss
                loss = self._compute_medical_loss(outputs, batch)
                val_metrics['val_loss'] += loss.item()
                
                # Collect predictions and labels for metrics
                if 'classification' in outputs:
                    predictions = torch.sigmoid(outputs['classification'])
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(batch['labels']['binary'].cpu().numpy())
        
        # Calculate metrics
        if all_predictions:
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # Convert to binary predictions
            binary_preds = (all_predictions > 0.5).astype(int)
            
            # Calculate metrics for each class
            for i in range(all_predictions.shape[1]):
                try:
                    auc = roc_auc_score(all_labels[:, i], all_predictions[:, i])
                    val_metrics[f'val_auc_class_{i}'] = auc
                except:
                    val_metrics[f'val_auc_class_{i}'] = 0.0
                
                accuracy = accuracy_score(all_labels[:, i], binary_preds[:, i])
                val_metrics[f'val_accuracy_class_{i}'] = accuracy
            
            # Overall metrics
            val_metrics['val_loss'] /= len(dataloader)
            val_metrics['val_accuracy'] = accuracy_score(all_labels, binary_preds)
            
            try:
                val_metrics['val_auc'] = roc_auc_score(all_labels, all_predictions, average='macro')
            except:
                val_metrics['val_auc'] = 0.0
            
            val_metrics['val_f1'] = f1_score(all_labels, binary_preds, average='macro')
        
        return val_metrics
    
    def train(self, 
              model: nn.Module, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int) -> Dict[str, List[float]]:
        """Complete training loop"""
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_auc': []
        }
        
        best_val_auc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(model, train_loader, optimizer, scheduler)
            
            # Validation phase
            val_metrics = self.validate(model, val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_accuracy'].append(val_metrics['val_accuracy'])
            history['val_auc'].append(val_metrics['val_auc'])
            
            # Save best model
            if val_metrics['val_auc'] > best_val_auc:
                best_val_auc = val_metrics['val_auc']
                best_model_state = model.state_dict().copy()
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Accuracy: {val_metrics['val_accuracy']:.4f}, "
                f"Val AUC: {val_metrics['val_auc']:.4f}"
            )
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return history