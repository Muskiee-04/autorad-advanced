"""
Test script to verify all components work
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    modules = [
        'data.processors.dicom_processor',
        'data.processors.text_processor', 
        'models.encoders.fusion_networks',
        'models.generators.report_generator',
        'deployment.gradio_ui',
        'utils.config'
    ]
    
    print("Testing imports...")
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    print("\nTesting package availability...")
    packages = ['torch', 'transformers', 'gradio', 'pydicom', 'monai', 'spacy']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError as e:
            print(f"❌ {pkg}: {e}")

if __name__ == "__main__":
    test_imports()