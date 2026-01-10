#!/bin/bash
# GeometryCrafter Low-Memory Pipeline Orchestrator
# This script runs the inference pipeline in isolated steps to minimize VRAM usage

set -e  # Exit on error

# Parse arguments
VIDEO_PATH=""
OUTPUT_DIR="workspace/output"
CACHE_DIR="workspace/cache"
TEMP_DIR="workspace/temp"
HEIGHT=""
WIDTH=""
DOWNSAMPLE_RATIO=1.0
NUM_INFERENCE_STEPS=5
GUIDANCE_SCALE=1.0
WINDOW_SIZE=110
DECODE_CHUNK_SIZE=8
OVERLAP=25
PROCESS_LENGTH=-1
PROCESS_STRIDE=1
SEED=42
MODEL_TYPE="diff"
FORCE_PROJECTION=true
FORCE_FIXED_FOCAL=true
USE_EXTRACT_INTERP=false

# Display usage
usage() {
    echo "Usage: $0 --video_path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --video_path PATH              Input video file path"
    echo ""
    echo "Optional:"
    echo "  --output_dir DIR               Output directory (default: workspace/output)"
    echo "  --cache_dir DIR                Model cache directory (default: workspace/cache)"
    echo "  --temp_dir DIR                 Temporary intermediate files (default: workspace/temp)"
    echo "  --height H                     Target height (must be divisible by 64)"
    echo "  --width W                      Target width (must be divisible by 64)"
    echo "  --downsample_ratio RATIO       Downsample ratio (default: 1.0)"
    echo "  --num_inference_steps N        Denoising steps (default: 5)"
    echo "  --guidance_scale SCALE         Guidance scale (default: 1.0)"
    echo "  --window_size SIZE             Processing window size (default: 110)"
    echo "  --decode_chunk_size SIZE       Chunk size for decoding (default: 8)"
    echo "  --overlap SIZE                 Overlap between windows (default: 25)"
    echo "  --process_length LEN           Number of frames to process (default: -1, all)"
    echo "  --process_stride STRIDE        Frame stride (default: 1)"
    echo "  --seed SEED                    Random seed (default: 42)"
    echo "  --model_type TYPE              Model type: 'diff' or 'determ' (default: diff)"
    echo "  --no_force_projection          Disable forced projection"
    echo "  --no_force_fixed_focal         Disable fixed focal length"
    echo "  --use_extract_interp           Use exact interpolation"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video_path) VIDEO_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --cache_dir) CACHE_DIR="$2"; shift 2 ;;
        --temp_dir) TEMP_DIR="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --downsample_ratio) DOWNSAMPLE_RATIO="$2"; shift 2 ;;
        --num_inference_steps) NUM_INFERENCE_STEPS="$2"; shift 2 ;;
        --guidance_scale) GUIDANCE_SCALE="$2"; shift 2 ;;
        --window_size) WINDOW_SIZE="$2"; shift 2 ;;
        --decode_chunk_size) DECODE_CHUNK_SIZE="$2"; shift 2 ;;
        --overlap) OVERLAP="$2"; shift 2 ;;
        --process_length) PROCESS_LENGTH="$2"; shift 2 ;;
        --process_stride) PROCESS_STRIDE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --no_force_projection) FORCE_PROJECTION=false; shift ;;
        --no_force_fixed_focal) FORCE_FIXED_FOCAL=false; shift ;;
        --use_extract_interp) USE_EXTRACT_INTERP=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: --video_path is required"
    usage
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$TEMP_DIR"

# Get video basename for output naming
VIDEO_BASENAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')

echo "========================================"
echo "GeometryCrafter Low-Memory Pipeline"
echo "========================================"
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "Temp: $TEMP_DIR"
echo "Model Type: $MODEL_TYPE"
echo "========================================"

# Define intermediate directory paths (per-frame mode)
PRIORS_DIR="$TEMP_DIR/${VIDEO_BASENAME}_priors"
CONTEXT_DIR="$TEMP_DIR/${VIDEO_BASENAME}_context"
DENOISED_FILE="$TEMP_DIR/${VIDEO_BASENAME}_denoised.pt"
VIDEO_INFO_FILE="$TEMP_DIR/${VIDEO_BASENAME}_video_info.pt"

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up intermediate files..."
    rm -rf "$PRIORS_DIR" "$CONTEXT_DIR"
    rm -f "$DENOISED_FILE" "$VIDEO_INFO_FILE"
    echo "Cleanup complete."
}

# Register cleanup on exit
trap cleanup EXIT

# Step 1: Generate Geometry Priors (Per-Frame)
echo ""
echo "==> Step 1/4: Computing Geometry Priors (MoGe) - Per-Frame Mode"
python bin/step1_priors.py \
    --video_path "$VIDEO_PATH" \
    --output_dir "$PRIORS_DIR" \
    --video_info_path "$VIDEO_INFO_FILE" \
    --cache_dir "$CACHE_DIR" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --downsample_ratio "$DOWNSAMPLE_RATIO" \
    --process_length "$PROCESS_LENGTH" \
    --process_stride "$PROCESS_STRIDE"

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed"
    exit 1
fi

# Step 2: Encode Context (Per-Frame: Image Embeddings + VAE Latents + Prior Latents)
echo ""
echo "==> Step 2/4: Encoding Context (Image Encoder + VAE + PointMapVAE) - Per-Frame Mode"
python bin/step2_encode.py \
    --video_path "$VIDEO_PATH" \
    --priors_dir "$PRIORS_DIR" \
    --video_info_path "$VIDEO_INFO_FILE" \
    --output_dir "$CONTEXT_DIR" \
    --cache_dir "$CACHE_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed"
    exit 1
fi

# Step 3: Denoising (UNet) - Per-Frame Context Loading
echo ""
echo "==> Step 3/4: Denoising Latents (UNet - VRAM intensive) - Per-Frame Mode"
python bin/step3_denoise.py \
    --context_dir "$CONTEXT_DIR" \
    --video_info_path "$VIDEO_INFO_FILE" \
    --output_path "$DENOISED_FILE" \
    --cache_dir "$CACHE_DIR" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --window_size "$WINDOW_SIZE" \
    --overlap "$OVERLAP" \
    --seed "$SEED" \
    --model_type "$MODEL_TYPE"

if [ $? -ne 0 ]; then
    echo "Error: Step 3 failed"
    exit 1
fi

# Step 4: Decode to Geometry Maps
echo ""
echo "==> Step 4/4: Decoding to Geometry Maps (PointMapVAE)"
python bin/step4_decode.py \
    --denoised_path "$DENOISED_FILE" \
    --video_info_path "$VIDEO_INFO_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --video_basename "$VIDEO_BASENAME" \
    --cache_dir "$CACHE_DIR" \
    --decode_chunk_size "$DECODE_CHUNK_SIZE" \
    --force_projection "$FORCE_PROJECTION" \
    --force_fixed_focal "$FORCE_FIXED_FOCAL" \
    --use_extract_interp "$USE_EXTRACT_INTERP"

if [ $? -ne 0 ]; then
    echo "Error: Step 4 failed"
    exit 1
fi

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "Output saved to: $OUTPUT_DIR/${VIDEO_BASENAME}.npz"
echo "========================================"
