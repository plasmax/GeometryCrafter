#!/bin/bash
# Cleanup temporary intermediate files from the low-memory pipeline

TEMP_DIR="${1:-workspace/temp}"

if [ ! -d "$TEMP_DIR" ]; then
    echo "Temp directory not found: $TEMP_DIR"
    exit 0
fi

echo "Cleaning up temporary files in $TEMP_DIR..."

# Count files
PRIORS=$(find "$TEMP_DIR" -name "*_priors.pt" 2>/dev/null | wc -l)
CONTEXT=$(find "$TEMP_DIR" -name "*_context.pt" 2>/dev/null | wc -l)
DENOISED=$(find "$TEMP_DIR" -name "*_denoised.pt" 2>/dev/null | wc -l)
INFO=$(find "$TEMP_DIR" -name "*_video_info.pt" 2>/dev/null | wc -l)

TOTAL=$((PRIORS + CONTEXT + DENOISED + INFO))

if [ $TOTAL -eq 0 ]; then
    echo "No temporary files found."
    exit 0
fi

echo "Found:"
echo "  - $PRIORS priors files"
echo "  - $CONTEXT context files"
echo "  - $DENOISED denoised files"
echo "  - $INFO info files"
echo "  Total: $TOTAL files"
echo ""

# Calculate total size
if command -v du &> /dev/null; then
    SIZE=$(du -sh "$TEMP_DIR" 2>/dev/null | cut -f1)
    echo "Total size: $SIZE"
    echo ""
fi

read -p "Delete all temporary files? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -f "$TEMP_DIR"/*_priors.pt
    rm -f "$TEMP_DIR"/*_context.pt
    rm -f "$TEMP_DIR"/*_denoised.pt
    rm -f "$TEMP_DIR"/*_video_info.pt
    echo "✓ Cleanup complete!"
else
    echo "Cleanup cancelled."
fi
