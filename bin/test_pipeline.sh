#!/bin/bash
# Test script for the low-memory pipeline
# This script validates that all steps can be executed

set -e

echo "========================================"
echo "GeometryCrafter Pipeline Test"
echo "========================================"
echo ""
echo "This script will test the low-memory pipeline setup."
echo "It does NOT run inference - it only validates the scripts."
echo ""

# Check if bin directory exists
if [ ! -d "bin" ]; then
    echo "Error: bin/ directory not found. Run this from the GeometryCrafter root."
    exit 1
fi

# Check all scripts exist
SCRIPTS=(
    "bin/run_geocrafter.sh"
    "bin/step1_priors.py"
    "bin/step2_encode.py"
    "bin/step3_denoise.py"
    "bin/step4_decode.py"
)

echo "Checking script files..."
for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "  ✗ Missing: $script"
        exit 1
    else
        echo "  ✓ Found: $script"
    fi
done

echo ""
echo "Checking script syntax..."

# Test Python scripts for syntax errors
for script in bin/step*.py; do
    if python -m py_compile "$script" 2>/dev/null; then
        echo "  ✓ $script - syntax OK"
    else
        echo "  ✗ $script - syntax error"
        exit 1
    fi
done

# Make scripts executable
echo ""
echo "Setting execute permissions..."
chmod +x bin/run_geocrafter.sh bin/test_pipeline.sh 2>/dev/null || echo "  (Windows - skipping chmod)"

echo ""
echo "Checking Python dependencies..."
python -c "
import sys
try:
    import torch
    print('  ✓ PyTorch:', torch.__version__)
except ImportError:
    print('  ✗ PyTorch not found')
    sys.exit(1)

try:
    import diffusers
    print('  ✓ Diffusers:', diffusers.__version__)
except ImportError:
    print('  ✗ Diffusers not found')
    sys.exit(1)

try:
    import decord
    print('  ✓ Decord installed')
except ImportError:
    print('  ✗ Decord not found')
    sys.exit(1)

try:
    import kornia
    print('  ✓ Kornia installed')
except ImportError:
    print('  ✗ Kornia not found')
    sys.exit(1)

try:
    import transformers
    print('  ✓ Transformers:', transformers.__version__)
except ImportError:
    print('  ✗ Transformers not found')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "Some dependencies are missing. Install them with:"
    echo "  pip install torch diffusers transformers decord kornia"
    exit 1
fi

echo ""
echo "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print('  ✓ CUDA available')
    print('  ✓ GPU:', torch.cuda.get_device_name(0))
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  ✓ VRAM: {mem_gb:.1f} GB')
    if mem_gb < 16:
        print('  ⚠ Warning: Less than 16GB VRAM detected')
else:
    print('  ✗ CUDA not available')
    print('  Note: CPU inference is not supported')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All checks passed!"
echo "========================================"
echo ""
echo "The low-memory pipeline is ready to use."
echo ""
echo "Example usage:"
echo "  ./bin/run_geocrafter.sh --video_path path/to/video.mp4"
echo ""
echo "See bin/README.md for full documentation."
