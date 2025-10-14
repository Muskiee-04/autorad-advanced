"""
Advanced medical text processing for radiology reports
"""
import re
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
import logging
import spacy
from spacy.tokens import DocBin
import pandas as pd

class MedicalTextProcessor:
    """Medical text processing with clinical NLP"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('text_backbone', 'emilyalsentzer/Bio_ClinicalBERT')
        )
        
        # Initialize clinical NLP
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except OSError:
            self.logger.warning("Clinical spaCy model not found, using small English model")
            self.nlp = spacy.load("en_core_web_sm")
    
    def process_radiology_report(self, report_text: str) -> Dict:
        """Process radiology report with clinical NLP"""
        # Clean text
        cleaned_text = self._clean_medical_text(report_text)
        
        # Tokenize for model
        tokenized = self.tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=self.config.get('text_max_length', 512),
            return_tensors='pt'
        )
        
        # Extract clinical entities
        clinical_entities = self._extract_clinical_entities(cleaned_text)
        
        # Extract measurements
        measurements = self._extract_measurements(cleaned_text)
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'clinical_entities': clinical_entities,
            'measurements': measurements,
            'cleaned_text': cleaned_text
        }
    
    def _clean_medical_text(self, text: str) -> str:
        """Clean medical text while preserving clinical information"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common headers
        headers = ['FINAL REPORT', 'INDICATION:', 'TECHNIQUE:', 'COMPARISON:', 'FINDINGS:', 'IMPRESSION:']
        for header in headers:
            text = text.replace(header, '')
        
        # Remove personal information patterns
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)  # Names
        text = re.sub(r'\b\d{1,3}[-/]?[A-Za-z]{0,10}[-/]?\d{1,4}\b', '[ID]', text)  # IDs
        
        return text.strip()
    
    def _extract_clinical_entities(self, text: str) -> List[Dict]:
        """Extract clinical entities using NLP"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def _extract_measurements(self, text: str) -> List[Dict]:
        """Extract measurements from radiology text"""
        measurements = []
        
        # Pattern for measurements like "5mm", "3.2 cm", "1.5 centimeter"
        patterns = [
            r'(\d+\.?\d*)\s*(mm|cm|millimeter|centimeter)',
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(mm|cm)',
            r'measuring\s*(\d+\.?\d*)\s*(mm|cm)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                measurements.append({
                    'value': float(match.group(1)),
                    'unit': match.group(2),
                    'text': match.group(0),
                    'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        
        return measurements
    
    def create_comparison_prompt(self, prior_report: str, current_findings: str) -> str:
        """Create clinical comparison prompt"""
        prompt = f"""
CLINICAL COMPARISON REQUEST:

PRIOR STUDY FINDINGS:
{prior_report}

CURRENT STUDY FINDINGS:
{current_findings}

Please provide a comprehensive radiology comparison report focusing on:
1. Interval changes (new findings, resolution, progression)
2. Stability of previous abnormalities
3. Quantitative measurements comparison
4. Clinical significance assessment
5. Recommendations for follow-up if indicated

COMPARISON REPORT:
"""
        return prompt.strip()