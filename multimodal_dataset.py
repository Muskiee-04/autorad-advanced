"""
Advanced multimodal dataset for medical AI
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import re

class MultimodalMedicalDataset(Dataset):
    """Multimodal dataset for medical image and text processing"""
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 image_dir: Path,
                 text_processor,
                 image_processor,
                 config: Dict,
                 is_training: bool = True):
        
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.config = config
        self.is_training = is_training
        
        # Image augmentations
        self.transform = self._get_transforms()
        
        self.logger = logging.getLogger(__name__)
    
    def _get_transforms(self):
        """Get image transformations"""
        if self.is_training:
            return A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=5,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        try:
            row = self.dataframe.iloc[idx]
            
            # Load and process image
            image_path = self.image_dir / row['dicom_id']
            image = self._load_image(image_path)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image=image)['image']
            
            # Process text
            report_text = row.get('report_text', '')
            text_data = self.text_processor.process_radiology_report(report_text)
            
            # Create labels
            labels = self._create_labels(row, report_text)
            
            return {
                'image': image,
                'input_ids': text_data['input_ids'].squeeze(),
                'attention_mask': text_data['attention_mask'].squeeze(),
                'labels': labels,
                'measurements': text_data['measurements'],
                'clinical_entities': text_data['clinical_entities'],
                'image_path': str(image_path),
                'report_text': report_text
            }
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load medical image"""
        if not image_path.exists():
            # Create dummy image if file doesn't exist
            return np.random.rand(224, 224).astype(np.float32)
        
        try:
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                # Process DICOM
                image = self.image_processor.preprocess_single_image(image_path)
                return image.squeeze().numpy()
            else:
                # Process regular image
                image = Image.open(image_path).convert('L')
                return np.array(image)
        except Exception as e:
            self.logger.warning(f"Error loading image {image_path}: {e}")
            return np.random.rand(224, 224).astype(np.float32)
    
    def _create_labels(self, row: pd.Series, report_text: str) -> Dict[str, torch.Tensor]:
        """Create multi-task labels"""
        text_lower = report_text.lower()
        
        # Binary classification labels
        labels = {
            'pneumonia': 1.0 if any(word in text_lower for word in ['pneumonia', 'consolidation']) else 0.0,
            'nodule': 1.0 if 'nodule' in text_lower else 0.0,
            'effusion': 1.0 if 'effusion' in text_lower else 0.0,
            'cardiomegaly': 1.0 if 'cardiomegaly' in text_lower else 0.0,
            'atelectasis': 1.0 if 'atelectasis' in text_lower else 0.0,
        }
        
        # Convert to tensor
        label_tensor = torch.tensor([labels[k] for k in sorted(labels.keys())], dtype=torch.float32)
        
        return {
            'binary': label_tensor,
            'measurements': torch.tensor(self._extract_numeric_measurements(report_text), dtype=torch.float32)
        }
    
    def _extract_numeric_measurements(self, text: str) -> List[float]:
        """Extract numeric measurements from text"""
        measurements = []
        patterns = [r'(\d+\.?\d*)\s*mm', r'(\d+\.?\d*)\s*cm']
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            measurements.extend([float(match) for match in matches])
        
        # Pad or truncate to fixed size
        max_measurements = 5
        if len(measurements) > max_measurements:
            measurements = measurements[:max_measurements]
        else:
            measurements.extend([0.0] * (max_measurements - len(measurements)))
        
        return measurements