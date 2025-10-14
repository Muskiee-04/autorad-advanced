"""
Advanced DICOM processing with medical-grade preprocessing
"""
import pydicom
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import SimpleITK as sitk
from pathlib import Path
import logging
import cv2
from skimage import exposure

class MedicalImageProcessor:
    """Medical-grade DICOM/medical image processing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_dicom_series(self, directory: Path) -> torch.Tensor:
        """Load and process DICOM series into 3D volume"""
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(str(directory))
            
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {directory}")
                
            reader.SetFileNames(dicom_files)
            image_3d = reader.Execute()
            
            # Convert to numpy and normalize
            image_array = sitk.GetArrayFromImage(image_3d)
            
            # Medical image normalization (CT/HU values)
            if self.config.get('hu_normalization', True):
                image_array = self._normalize_hu_values(image_array)
            
            # Resize if needed
            if self.config.get('image_size'):
                target_size = self.config['image_size']
                image_array = self._resize_volume(image_array, target_size)
            
            return torch.from_numpy(image_array).float()
            
        except Exception as e:
            self.logger.error(f"Error loading DICOM series: {e}")
            raise
    
    def _normalize_hu_values(self, image: np.ndarray) -> np.ndarray:
        """Normalize Hounsfield Units for medical imaging"""
        hu_min = self.config.get('hu_min', -1000)
        hu_max = self.config.get('hu_max', 2000)
        
        image = np.clip(image, hu_min, hu_max)
        image = (image - hu_min) / (hu_max - hu_min)
        return image
    
    def _resize_volume(self, volume: np.ndarray, target_size: List[int]) -> np.ndarray:
        """Resize 3D volume to target size"""
        # Simple resizing for demo - in production use proper 3D resampling
        if len(volume.shape) == 3:
            resized = np.zeros((volume.shape[0], target_size[0], target_size[1]))
            for i in range(volume.shape[0]):
                resized[i] = cv2.resize(volume[i], (target_size[1], target_size[0]))
            return resized
        return volume
    
    def extract_medical_metadata(self, dicom_path: Path) -> Dict:
        """Extract comprehensive medical metadata from DICOM"""
        try:
            ds = pydicom.dcmread(str(dicom_path))
            
            metadata = {
                'patient_id': getattr(ds, 'PatientID', ''),
                'study_date': getattr(ds, 'StudyDate', ''),
                'modality': getattr(ds, 'Modality', ''),
                'body_part': getattr(ds, 'BodyPartExamined', ''),
                'study_description': getattr(ds, 'StudyDescription', ''),
                'series_description': getattr(ds, 'SeriesDescription', ''),
                'slice_thickness': getattr(ds, 'SliceThickness', 0),
                'pixel_spacing': getattr(ds, 'PixelSpacing', [1, 1]),
                'kvp': getattr(ds, 'KVP', 0),
                'exposure': getattr(ds, 'Exposure', 0),
            }
            
            return metadata
        except Exception as e:
            self.logger.warning(f"Error extracting DICOM metadata: {e}")
            return {}
    
    def preprocess_single_image(self, image_path: Path) -> torch.Tensor:
        """Preprocess single medical image"""
        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            # DICOM file
            ds = pydicom.dcmread(str(image_path))
            image = ds.pixel_array
        else:
            # Regular image file
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply medical image preprocessing
        image = self._enhance_medical_image(image)
        
        # Resize
        target_size = self.config.get('image_size', (224, 224))
        image = cv2.resize(image, target_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
    
    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance medical image for better analysis"""
        # Ensure image is uint8 for CLAHE
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # Gamma correction
        image = exposure.adjust_gamma(image, gamma=0.8)
        
        return image