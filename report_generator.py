"""
Advanced radiology report generator with clinical knowledge integration
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GenerationConfig
)
from typing import Dict, List, Optional
import logging
import re

class ClinicalReportGenerator(nn.Module):
    """AI-powered radiology report generator with medical knowledge"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Medical LLM for report generation
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config['llm_model'],
                torch_dtype=torch.float16 if config.get('use_fp16', False) else torch.float32,
                device_map="auto" if config.get('use_device_map', False) else None
            )
        except Exception as e:
            self.logger.warning(f"Could not load {config['llm_model']}, using smaller model")
            self.llm = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['llm_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Medical knowledge prompts
        self.clinical_prompts = self._load_clinical_prompts()
        
        # Report structure templates
        self.report_templates = self._load_report_templates()
    
    def _load_clinical_prompts(self) -> Dict:
        """Load clinical prompting templates"""
        return {
            'comparison': """
You are an expert radiologist. Compare the current imaging findings with prior studies.

PRIOR FINDINGS:
{prior_findings}

CURRENT FINDINGS:
{current_findings}

IMAGE ANALYSIS:
{image_features}

CLINICAL CONTEXT:
{clinical_context}

Generate a structured radiology report comparing current and prior studies. Focus on:
1. Interval changes (size, appearance, new findings)
2. Stability of previous findings
3. Clinical significance of changes
4. Quantitative measurements comparison
5. Recommendations if needed

Format the report professionally using standard radiology structure.

REPORT:
""",
            'structured': """
Generate a structured radiology report using the following template:

EXAM: {modality} {body_part}
COMPARISON: {comparison_date}
CLINICAL HISTORY: {clinical_history}
TECHNIQUE: {technique}
FINDINGS:
{findings}
IMPRESSION:
{impression}
RECOMMENDATIONS:
{recommendations}
"""
        }
    
    def _load_report_templates(self) -> Dict:
        """Load radiology report templates"""
        return {
            'chest_xray': {
                'sections': ['FINDINGS', 'IMPRESSION', 'RECOMMENDATIONS'],
                'required_terms': ['lungs', 'heart', 'mediastinum', 'bones'],
                'measurement_units': ['mm', 'cm']
            },
            'ct_chest': {
                'sections': ['TECHNIQUE', 'COMPARISON', 'FINDINGS', 'IMPRESSION'],
                'required_terms': ['lung', 'nodule', 'vessels', 'mediastinum'],
                'measurement_units': ['mm', 'cm']
            }
        }
    
    def generate_report(self, 
                       image_features: torch.Tensor,
                       prior_text: str,
                       current_findings: str,
                       clinical_context: str = "",
                       generation_config: Optional[Dict] = None) -> Dict:
        """Generate clinical radiology report"""
        
        # Prepare prompt with clinical context
        prompt = self.clinical_prompts['comparison'].format(
            prior_findings=prior_text,
            current_findings=current_findings,
            image_features=self._format_image_features(image_features),
            clinical_context=clinical_context
        )
        
        # Medical-grade generation config
        if generation_config is None:
            generation_config = {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                'no_repeat_ngram_size': 3
            }
        
        # Generate report
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                **generation_config
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Extract structured components
        structured_report = self._extract_structured_components(generated_text)
        
        # Calculate clinical quality metrics
        quality_metrics = self._calculate_quality_metrics(generated_text, structured_report)
        
        return {
            'full_report': generated_text,
            'structured': structured_report,
            'quality_metrics': quality_metrics,
            'confidence': self._calculate_confidence(generated_text, structured_report)
        }
    
    def _format_image_features(self, features: torch.Tensor) -> str:
        """Convert image features to clinical descriptions"""
        # This would involve a feature-to-text model in production
        # For now, generate descriptive text based on feature patterns
        
        if features.dim() > 1:
            features = features.mean(dim=0)
        
        # Simple heuristic-based description
        descriptions = []
        
        # Analyze feature patterns (simplified)
        feature_mean = features.mean().item()
        feature_std = features.std().item()
        
        if feature_std > 0.5:
            descriptions.append("Image shows heterogeneous tissue patterns")
        else:
            descriptions.append("Image shows homogeneous tissue patterns")
        
        if feature_mean > 0:
            descriptions.append("Overall increased density noted")
        else:
            descriptions.append("Overall normal density pattern")
        
        return ". ".join(descriptions)
    
    def _extract_structured_components(self, report: str) -> Dict:
        """Extract structured components from generated report"""
        structured = {
            'findings': self._extract_section(report, 'FINDINGS'),
            'impression': self._extract_section(report, 'IMPRESSION'),
            'recommendations': self._extract_section(report, 'RECOMMENDATIONS'),
            'measurements': self._extract_measurements(report),
            'critical_findings': self._extract_critical_findings(report)
        }
        
        return structured
    
    def _extract_section(self, report: str, section_name: str) -> str:
        """Extract specific section from report"""
        pattern = f"{section_name}:(.*?)(?=\\n[A-Z]+:|$)"
        matches = re.findall(pattern, report, re.IGNORECASE | re.DOTALL)
        return matches[0].strip() if matches else ""
    
    def _extract_measurements(self, report: str) -> List[Dict]:
        """Extract measurements from report"""
        measurements = []
        
        # Pattern for various measurement formats
        patterns = [
            r'(\d+\.?\d*)\s*(mm|cm)\s*(?:x\s*(\d+\.?\d*)\s*(mm|cm))?',  # Size measurements
            r'(\d+\.?\d*)\s*(mm|cm)\s*(?:nodule|mass|lesion|opacity)',
            r'measuring\s*(\d+\.?\d*)\s*(mm|cm)',
            r'size[:\s]*(\d+\.?\d*)\s*(mm|cm)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, report, re.IGNORECASE)
            for match in matches:
                measurement = {
                    'value': float(match.group(1)),
                    'unit': match.group(2),
                    'text': match.group(0),
                    'context': report[max(0, match.start()-50):min(len(report), match.end()+50)]
                }
                
                # Handle 2D measurements
                if match.group(3) and match.group(4):
                    measurement['value2'] = float(match.group(3))
                    measurement['unit2'] = match.group(4)
                
                measurements.append(measurement)
        
        return measurements
    
    def _extract_critical_findings(self, report: str) -> List[str]:
        """Extract critical findings that need immediate attention"""
        critical_keywords = [
            'pneumothorax', 'hemorrhage', 'embolism', 'fracture', 'rupture',
            'obstruction', 'ischemia', 'infarction', 'abscess', 'effusion'
        ]
        
        critical_findings = []
        sentences = re.split(r'[.!?]+', report)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in critical_keywords):
                # Check for severity indicators
                severity_indicators = ['large', 'massive', 'significant', 'critical', 'emergent']
                if any(indicator in sentence.lower() for indicator in severity_indicators):
                    critical_findings.append(sentence.strip())
        
        return critical_findings
    
    def _calculate_quality_metrics(self, report: str, structured: Dict) -> Dict:
        """Calculate report quality metrics"""
        metrics = {}
        
        # Completeness
        sections_present = sum(1 for section in ['findings', 'impression'] if structured.get(section))
        metrics['completeness'] = sections_present / 2.0
        
        # Measurement presence
        metrics['measurements_present'] = len(structured.get('measurements', [])) > 0
        
        # Critical findings identification
        metrics['critical_findings_identified'] = len(structured.get('critical_findings', [])) > 0
        
        # Report length adequacy
        word_count = len(report.split())
        metrics['length_adequate'] = 50 <= word_count <= 1000
        
        return metrics
    
    def _calculate_confidence(self, report: str, structured: Dict) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # Length-based confidence
        word_count = len(report.split())
        if 100 <= word_count <= 800:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Structure-based confidence
        if structured.get('findings') and structured.get('impression'):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Measurement confidence
        if structured.get('measurements'):
            confidence_factors.append(0.7)
        
        # Critical findings confidence
        if structured.get('critical_findings'):
            confidence_factors.append(0.8)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

class ComparativeReportGenerator(ClinicalReportGenerator):
    """Specialized generator for comparative studies"""
    
    def generate_comparison_report(self,
                                 current_image_features: torch.Tensor,
                                 prior_image_features: torch.Tensor,
                                 prior_report: str,
                                 current_findings: str,
                                 clinical_context: str = "") -> Dict:
        """Generate comparative radiology report"""
        
        # Create comparative prompt
        prompt = f"""
RADIOLOGY COMPARISON REQUEST:

PRIOR STUDY REPORT:
{prior_report}

CURRENT STUDY FINDINGS:
{current_findings}

PRIOR IMAGE ANALYSIS:
{self._format_image_features(prior_image_features)}

CURRENT IMAGE ANALYSIS:
{self._format_image_features(current_image_features)}

CLINICAL CONTEXT:
{clinical_context}

Generate a detailed comparison report highlighting:
1. Significant interval changes
2. Stability or progression of known abnormalities
3. New findings in current study
4. Resolution of previous findings
5. Quantitative changes in measurements
6. Clinical implications

Focus on actionable insights and clear comparison metrics.

COMPARATIVE REPORT:
"""
        
        # Generate with conservative settings for accuracy
        generation_config = {
            'max_new_tokens': 600,
            'temperature': 0.5,  # Lower temperature for more consistent output
            'do_sample': True,
            'top_p': 0.85,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 4
        }
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, **generation_config)
        
        comparative_report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract comparison-specific structure
        structured_comparison = self._extract_comparison_components(comparative_report)
        
        return {
            'comparative_report': comparative_report,
            'structured_comparison': structured_comparison,
            'change_analysis': self._analyze_changes(prior_report, current_findings),
            'confidence': self._calculate_comparison_confidence(comparative_report, structured_comparison)
        }
    
    def _extract_comparison_components(self, report: str) -> Dict:
        """Extract comparison-specific components"""
        components = {
            'interval_changes': self._extract_interval_changes(report),
            'stable_findings': self._extract_stable_findings(report),
            'new_findings': self._extract_new_findings(report),
            'resolved_findings': self._extract_resolved_findings(report),
            'quantitative_changes': self._extract_quantitative_changes(report)
        }
        return components
    
    def _extract_interval_changes(self, report: str) -> List[str]:
        """Extract described interval changes"""
        change_indicators = [
            'increased', 'decreased', 'enlarged', 'reduced', 'progressed', 'improved',
            'worsened', 'grown', 'resolved', 'developed', 'appeared', 'disappeared'
        ]
        
        changes = []
        sentences = re.split(r'[.!?]+', report)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in change_indicators):
                changes.append(sentence.strip())
        
        return changes
    
    def _extract_stable_findings(self, report: str) -> List[str]:
        """Extract findings described as stable"""
        stable_indicators = ['stable', 'unchanged', 'no change', 'remains']
        
        stable_findings = []
        sentences = re.split(r'[.!?]+', report)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in stable_indicators):
                stable_findings.append(sentence.strip())
        
        return stable_findings
    
    def _extract_new_findings(self, report: str) -> List[str]:
        """Extract new findings"""
        new_indicators = ['new', 'not previously seen', 'not on prior']
        
        new_findings = []
        sentences = re.split(r'[.!?]+', report)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in new_indicators):
                new_findings.append(sentence.strip())
        
        return new_findings
    
    def _extract_resolved_findings(self, report: str) -> List[str]:
        """Extract resolved findings"""
        resolved_indicators = ['resolved', 'resolution', 'no longer seen', 'cleared']
        
        resolved_findings = []
        sentences = re.split(r'[.!?]+', report)
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in resolved_indicators):
                resolved_findings.append(sentence.strip())
        
        return resolved_findings
    
    def _extract_quantitative_changes(self, report: str) -> List[Dict]:
        """Extract quantitative changes with measurements"""
        quantitative_changes = []
        
        # Pattern for quantitative comparisons
        patterns = [
            r'from\s*(\d+\.?\d*)\s*(mm|cm)\s*to\s*(\d+\.?\d*)\s*(mm|cm)',
            r'increased\s*from\s*(\d+\.?\d*)\s*(mm|cm)\s*to\s*(\d+\.?\d*)\s*(mm|cm)',
            r'decreased\s*from\s*(\d+\.?\d*)\s*(mm|cm)\s*to\s*(\d+\.?\d*)\s*(mm|cm)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, report, re.IGNORECASE)
            for match in matches:
                change = {
                    'previous_value': float(match.group(1)),
                    'previous_unit': match.group(2),
                    'current_value': float(match.group(3)),
                    'current_unit': match.group(4),
                    'change': float(match.group(3)) - float(match.group(1)),
                    'change_percent': ((float(match.group(3)) - float(match.group(1))) / float(match.group(1))) * 100,
                    'text': match.group(0)
                }
                quantitative_changes.append(change)
        
        return quantitative_changes
    
    def _analyze_changes(self, prior_report: str, current_findings: str) -> Dict:
        """Analyze changes between prior and current findings"""
        # Simple keyword-based change analysis
        prior_lower = prior_report.lower()
        current_lower = current_findings.lower()
        
        # Common radiology findings
        findings = ['nodule', 'mass', 'opacity', 'effusion', 'consolidation', 'fibrosis']
        
        change_analysis = {}
        for finding in findings:
            in_prior = finding in prior_lower
            in_current = finding in current_lower
            
            if in_prior and not in_current:
                change_analysis[finding] = 'resolved'
            elif not in_prior and in_current:
                change_analysis[finding] = 'new'
            elif in_prior and in_current:
                change_analysis[finding] = 'persistent'
            else:
                change_analysis[finding] = 'absent'
        
        return change_analysis
    
    def _calculate_comparison_confidence(self, report: str, structured_comparison: Dict) -> float:
        """Calculate confidence for comparison reports"""
        confidence_factors = []
        
        # Change analysis completeness
        change_categories = ['interval_changes', 'stable_findings', 'new_findings', 'resolved_findings']
        present_categories = sum(1 for category in change_categories if structured_comparison.get(category))
        confidence_factors.append(present_categories / len(change_categories))
        
        # Quantitative changes presence
        if structured_comparison.get('quantitative_changes'):
            confidence_factors.append(0.8)
        
        # Report structure
        if len(report.split()) > 200:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5