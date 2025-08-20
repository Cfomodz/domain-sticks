#!/bin/bash
# Activation script for domain-sticks project

echo "🚀 Activating domain-sticks virtual environment..."
source venv/bin/activate

echo "✅ Virtual environment activated!"
echo "🖥️  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "🎯 CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "🚀 GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo ""
    echo "🎬 Ready for GPU-accelerated video processing!"
else
    echo "⚠️  No GPU detected - will use CPU mode"
fi

echo ""
echo "To run your project, use: python src/driver.py"
echo "To deactivate, use: deactivate"
