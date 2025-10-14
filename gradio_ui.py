"""
Advanced Gradio interface for clinical deployment
"""
import gradio as gr
import torch
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import tempfile
import cv2
from PIL import Image
import json

class AutoRadInterface:
    """Clinical-grade interface for AutoRad system"""
    
    def __init__(self, model, processor, config):
        self.model = model
        self.processor = processor
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_study(self, 
                     current_image: np.ndarray,
                     prior_image: Optional[np.ndarray],
                     prior_report: str,
                     clinical_context: str) -> Dict:
        """Comprehensive study analysis"""
        
        try:
            # Process current image
            current_tensor = self.processor.preprocess_single_image(current_image)
            current_features = self.model.image_encoder(current_tensor.unsqueeze(0).to(self.device))
            
            # Process prior image if available
            prior_features = None
            if prior_image is not None:
                prior_tensor = self.processor.preprocess_single_image(prior_image)
                prior_features = self.model.image_encoder(prior_tensor.unsqueeze(0).to(self.device))
            
            # Process text data
            text_data = self.processor.text_processor.process_radiology_report(prior_report)
            
            # Generate analysis
            with torch.no_grad():
                if prior_features is not None:
                    # Comparative analysis
                    analysis = self.model.report_generator.generate_comparison_report(
                        current_features[0] if isinstance(current_features, tuple) else current_features,
                        prior_features[0] if isinstance(prior_features, tuple) else prior_features,
                        prior_report,
                        "Current imaging findings based on AI analysis",
                        clinical_context
                    )
                else:
                    # Single study analysis
                    analysis = self.model.report_generator.generate_report(
                        current_features[0] if isinstance(current_features, tuple) else current_features,
                        "",
                        "Current imaging findings based on AI analysis", 
                        clinical_context
                    )
            
            return {
                'comparative_report': analysis.get('comparative_report', analysis.get('full_report', '')),
                'critical_findings': analysis.get('structured', {}).get('critical_findings', []),
                'measurement_changes': analysis.get('structured_comparison', {}).get('quantitative_changes', []),
                'confidence_scores': analysis.get('confidence', 0.5),
                'recommendations': analysis.get('structured', {}).get('recommendations', '')
            }
            
        except Exception as e:
            return {
                'error': f"Analysis failed: {str(e)}",
                'comparative_report': '',
                'critical_findings': [],
                'measurement_changes': [],
                'confidence_scores': 0.0,
                'recommendations': ''
            }
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        def process_study(current_image, prior_image, prior_report, clinical_context):
            results = self.analyze_study(current_image, prior_image, prior_report, clinical_context)
            
            if 'error' in results:
                return results['error'], "", [], [], "", ""
            
            # Format critical findings
            critical_findings_formatted = []
            for finding in results['critical_findings']:
                critical_findings_formatted.append((finding, "CRITICAL"))
            
            # Format measurement changes
            measurements_formatted = []
            for measurement in results['measurement_changes']:
                measurements_formatted.append({
                    'previous': f"{measurement['previous_value']} {measurement['previous_unit']}",
                    'current': f"{measurement['current_value']} {measurement['current_unit']}",
                    'change': f"{measurement['change']:+.1f} {measurement['current_unit']}",
                    'change_percent': f"{measurement['change_percent']:+.1f}%"
                })
            
            return (
                results['comparative_report'],
                critical_findings_formatted,
                measurements_formatted,
                f"Confidence: {results['confidence_scores']:.2%}",
                results['recommendations'],
                ""  # Clear any previous errors
            )
        
        def clear_all():
            return None, None, "", "", "", [], {}, "", ""
        
        # Interface components
        with gr.Blocks(theme=gr.themes.Soft(), title="AutoRad Clinical") as demo:
            gr.Markdown("# 🏥 AutoRad Clinical - Advanced Radiology AI")
            gr.Markdown("### AI-Powered Comparative Radiology Reporting")
            
            with gr.Row():
                with gr.Column():
                    current_image = gr.Image(
                        label="Current Study Image",
                        type="numpy",
                        height=300
                    )
                    prior_image = gr.Image(
                        label="Prior Study Image (Optional)",
                        type="numpy", 
                        height=300
                    )
                    
                with gr.Column():
                    prior_report = gr.Textbox(
                        label="Prior Radiology Report",
                        lines=8,
                        placeholder="Paste the prior radiology report here...",
                        info="Include findings, measurements, and impression from previous study"
                    )
                    clinical_context = gr.Textbox(
                        label="Clinical Context",
                        lines=3,
                        placeholder="Relevant clinical history, symptoms, reason for current study...",
                        info="e.g., '65yo male with cough and fever, follow-up for known lung nodule'"
                    )
            
            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze Study", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear All", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    report_output = gr.Textbox(
                        label="Comparative Radiology Report",
                        lines=12,
                        interactive=False
                    )
                
                with gr.Column():
                    findings_output = gr.HighlightedText(
                        label="Critical Findings",
                        show_legend=True,
                        color_map={"CRITICAL": "red"}
                    )
                    confidence_output = gr.Label(
                        label="Analysis Confidence"
                    )
            
            with gr.Row():
                measurements_output = gr.JSON(
                    label="Quantitative Changes"
                )
                recommendations_output = gr.Textbox(
                    label="Clinical Recommendations",
                    lines=3,
                    interactive=False
                )
            
            error_output = gr.Textbox(
                label="Status",
                visible=False
            )
            
            # Examples
            with gr.Accordion("📋 Example Inputs", open=False):
                gr.Examples(
                    examples=[
                        [
                            np.random.rand(224, 224, 3).astype(np.float32),  # Dummy current image
                            np.random.rand(224, 224, 3).astype(np.float32),  # Dummy prior image
                            "CHEST X-RAY: Lungs are clear. Heart size normal. No pleural effusion. No pneumothorax.",
                            "45yo female with persistent cough, follow-up for known 5mm lung nodule"
                        ]
                    ],
                    inputs=[current_image, prior_image, prior_report, clinical_context]
                )
            
            # Button actions
            analyze_btn.click(
                fn=process_study,
                inputs=[current_image, prior_image, prior_report, clinical_context],
                outputs=[report_output, findings_output, measurements_output, confidence_output, recommendations_output, error_output]
            )
            
            clear_btn.click(
                fn=clear_all,
                inputs=[],
                outputs=[current_image, prior_image, prior_report, clinical_context, report_output, findings_output, measurements_output, confidence_output, recommendations_output]
            )
        
        return demo

def create_demo_interface():
    """Create a demo interface for testing without full model initialization"""
    
    def demo_analysis(current_image, prior_image, prior_report, clinical_context):
        # Mock analysis for demo purposes
        if current_image is None:
            return "Please upload a current study image", [], [], "Confidence: 0%", "", "Error: No image provided"
        
        # Simulate processing time
        import time
        time.sleep(2)
        
        # Mock report based on inputs
        report = f"""
COMPARATIVE RADIOLOGY REPORT

EXAM: Chest X-Ray PA and Lateral
COMPARISON: Prior study from 2023
CLINICAL HISTORY: {clinical_context if clinical_context else 'Not provided'}

FINDINGS:
LUNGS: Clear without focal consolidation. No evidence of pneumonia.
HEART: Normal cardiomediastinal silhouette.
PLEURA: No pleural effusion or pneumothorax.
BONES: No acute fracture.

IMPRESSION:
1. No acute cardiopulmonary process.
2. Stable appearance compared to prior study.
3. Recommend clinical correlation.

{'' if prior_image is None else '4. No significant interval change from prior examination.'}
"""

        critical_findings = []
        if "pneumonia" in prior_report.lower() or "effusion" in prior_report.lower():
            critical_findings = [("Stable pleural findings, continue monitoring", "CRITICAL")]
        
        measurements = []
        if "mm" in prior_report or "cm" in prior_report:
            measurements = [{
                'previous': '5.0 mm',
                'current': '5.2 mm', 
                'change': '+0.2 mm',
                'change_percent': '+4.0%'
            }]
        
        return report, critical_findings, measurements, "Confidence: 85%", "Continue routine follow-up as clinically indicated", ""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="AutoRad Demo") as demo:
        gr.Markdown("# 🏥 AutoRad Clinical - Demo Interface")
        gr.Markdown("### AI-Powered Comparative Radiology Reporting (Demo Mode)")
        
        with gr.Row():
            with gr.Column():
                current_image = gr.Image(
                    label="Current Study Image",
                    type="numpy",
                    height=300
                )
                prior_image = gr.Image(
                    label="Prior Study Image (Optional)",
                    type="numpy", 
                    height=300
                )
                
            with gr.Column():
                prior_report = gr.Textbox(
                    label="Prior Radiology Report",
                    lines=8,
                    placeholder="Paste the prior radiology report here...",
                    value="CHEST X-RAY: Lungs are clear. Heart size normal. No pleural effusion. Stable 5mm right lower lobe nodule."
                )
                clinical_context = gr.Textbox(
                    label="Clinical Context",
                    lines=3,
                    placeholder="Relevant clinical history, symptoms, reason for current study...",
                    value="65yo male with cough and fever, follow-up for known lung nodule"
                )
        
        with gr.Row():
            analyze_btn = gr.Button("🚀 Analyze Study", variant="primary", size="lg")
            clear_btn = gr.Button("🔄 Clear All", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                report_output = gr.Textbox(
                    label="Comparative Radiology Report",
                    lines=12,
                    interactive=False
                )
            
            with gr.Column():
                findings_output = gr.HighlightedText(
                    label="Critical Findings",
                    show_legend=True,
                    color_map={"CRITICAL": "red"}
                )
                confidence_output = gr.Label(
                    label="Analysis Confidence"
                )
        
        with gr.Row():
            measurements_output = gr.JSON(
                label="Quantitative Changes"
            )
            recommendations_output = gr.Textbox(
                label="Clinical Recommendations",
                lines=3,
                interactive=False
            )
        
        error_output = gr.Textbox(
            label="Status",
            visible=False
        )
        
        # Button actions
        analyze_btn.click(
            fn=demo_analysis,
            inputs=[current_image, prior_image, prior_report, clinical_context],
            outputs=[report_output, findings_output, measurements_output, confidence_output, recommendations_output, error_output]
        )
        
        clear_btn.click(
            fn=lambda: [None, None, "", "", "", [], {}, "", ""],
            inputs=[],
            outputs=[current_image, prior_image, prior_report, clinical_context, report_output, findings_output, measurements_output, confidence_output, recommendations_output]
        )
    
    return demo

if __name__ == "__main__":
    # Launch the demo interface
    demo = create_demo_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )