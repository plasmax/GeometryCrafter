#!/bin/bash
# GeometryCrafter Low-Memory Pipeline Orchestrator
# This script runs the inference pipeline in isolated steps to minimize VRAM usage

# Debug mode - set DEBUG_MODE=true to pause before cleanup on error
DEBUG_MODE="${DEBUG_MODE:-false}"

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
    echo "  --attention_mode MODE          Attention mode: auto, xformers, sdpa, slicing (default: auto)"
    echo "  --cpu_offload                  Enable sequential CPU offload for UNet"
    echo "  --residual_offload MODE        Residual offload mode: none, cpu, disk (default: cpu)"
    echo "  --residual_cache_dir DIR       Directory for residual disk cache (default: temp/residuals)"
    echo "  --enable_unet_checkpointing    Enable gradient checkpointing for UNet (reduces VRAM, may affect quality)"
    echo "  --no_vae_offloading            Disable VAE CPU offloading"
    echo "  --no_inference_checkpointing   Disable inference gradient checkpointing for VAE"
    echo "  --save_mp4                     Generate MP4 preview at the end (default: true)"
    echo "  --no_save_mp4                  Disable MP4 generation"
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
        --attention_mode) ATTENTION_MODE="$2"; shift 2 ;;
        --cpu_offload) CPU_OFFLOAD=true; shift ;;
        --residual_offload) RESIDUAL_OFFLOAD="$2"; shift 2 ;;
        --residual_cache_dir) RESIDUAL_CACHE_DIR="$2"; shift 2 ;;
        --enable_unet_checkpointing) ENABLE_UNET_CHECKPOINTING=true; shift ;;
        --no_force_projection) FORCE_PROJECTION=false; shift ;;
        --no_force_fixed_focal) FORCE_FIXED_FOCAL=false; shift ;;
        --use_extract_interp) USE_EXTRACT_INTERP=true; shift ;;
        --no_vae_offloading) ENABLE_VAE_OFFLOADING=false; shift ;;
        --no_inference_checkpointing) ENABLE_INFERENCE_CHECKPOINTING=false; shift ;;
        --save_mp4) SAVE_MP4=true; shift ;;
        --no_save_mp4) SAVE_MP4=false; shift ;;
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

# Initialize defaults for new arguments
ATTENTION_MODE="${ATTENTION_MODE:-auto}"
CPU_OFFLOAD="${CPU_OFFLOAD:-false}"
RESIDUAL_OFFLOAD="${RESIDUAL_OFFLOAD:-cpu}"  # Default to cpu as requested
RESIDUAL_CACHE_DIR="${RESIDUAL_CACHE_DIR:-$TEMP_DIR/residuals}"
ENABLE_UNET_CHECKPOINTING="${ENABLE_UNET_CHECKPOINTING:-false}"  # Default false for quality
ENABLE_VAE_OFFLOADING="${ENABLE_VAE_OFFLOADING:-true}"
ENABLE_INFERENCE_CHECKPOINTING="${ENABLE_INFERENCE_CHECKPOINTING:-true}"
SAVE_MP4="${SAVE_MP4:-true}"

# Display configuration
echo "========================================"
echo "GeometryCrafter Low-Memory Pipeline"
echo "========================================"
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "Temp: $TEMP_DIR"
echo "Model Type: $MODEL_TYPE"
echo "Memory Optimization:"
echo "  Attention: $ATTENTION_MODE"
echo "  CPU Offload: $CPU_OFFLOAD"
echo "  Residual Offload: $RESIDUAL_OFFLOAD"
echo "  Residual Cache: $RESIDUAL_CACHE_DIR"
echo "========================================"

# Define intermediate directory paths (per-frame mode)
PRIORS_DIR="$TEMP_DIR/${VIDEO_BASENAME}_priors"
CONTEXT_DIR="$TEMP_DIR/${VIDEO_BASENAME}_context"
DENOISED_FILE="$TEMP_DIR/${VIDEO_BASENAME}_denoised.pt"
VIDEO_INFO_FILE="$TEMP_DIR/${VIDEO_BASENAME}_video_info.pt"

# Create residual cache directory if needed
if [ "$RESIDUAL_OFFLOAD" = "disk" ]; then
    mkdir -p "$RESIDUAL_CACHE_DIR"
fi

# Cleanup function
cleanup() {
    if [ "$DEBUG_MODE" = "true" ]; then
        echo ""
        echo "========================================"
        echo "ERROR OCCURRED - Debug Mode Active"
        echo "========================================"
        echo "Intermediate files preserved in: $TEMP_DIR"
        echo "Residual cache preserved in: $RESIDUAL_CACHE_DIR"
        echo "Press ENTER to cleanup and exit, or Ctrl+C to keep files..."
        read -r
    fi
    echo ""
    echo "Cleaning up intermediate files..."
    rm -rf "$PRIORS_DIR" "$CONTEXT_DIR"
    
    # Clean up residual cache if it's within workspace/temp (or just generally if we created it)
    # To be safe, we only delete it if we are using disk offload and it is inside our control
    if [ "$RESIDUAL_OFFLOAD" = "disk" ] && [ -d "$RESIDUAL_CACHE_DIR" ]; then
        echo "Cleaning up residual cache..."
        rm -rf "$RESIDUAL_CACHE_DIR"
    fi
    
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
    --model_type "$MODEL_TYPE" \
    --attention_mode "$ATTENTION_MODE" \
    --residual_offload "$RESIDUAL_OFFLOAD" \
    --residual_cache_dir "$RESIDUAL_CACHE_DIR" \
    $( [ "$CPU_OFFLOAD" = "true" ] && echo "--cpu_offload" ) \
    $( [ "$ENABLE_UNET_CHECKPOINTING" = "true" ] && echo "--enable_unet_checkpointing" )

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
    --use_extract_interp "$USE_EXTRACT_INTERP" \
    --enable_vae_offloading "$ENABLE_VAE_OFFLOADING" \
    --enable_inference_checkpointing "$ENABLE_INFERENCE_CHECKPOINTING"

if [ $? -ne 0 ]; then
    echo "Error: Step 4 failed"
    exit 1
fi

# Step 5: Convert to MP4
if [ "$SAVE_MP4" = "true" ]; then
    echo ""
    echo "==> Step 5/5: Generating MP4 Preview"
    
    OUTPUT_NPZ="$OUTPUT_DIR/${VIDEO_BASENAME}.npz"
    OUTPUT_MP4="$OUTPUT_DIR/${VIDEO_BASENAME}.mp4"
    
    python bin/npz_to_mp4.py \
        --npz_path "$OUTPUT_NPZ" \
        --output_path "$OUTPUT_MP4" \
        --fps 30
        
    if [ $? -ne 0 ]; then
        echo "Warning: MP4 generation failed (but pipeline finished successfully)"
    else
        echo "MP4 saved to: $OUTPUT_MP4"
    fi
fi

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "Output saved to: $OUTPUT_DIR/${VIDEO_BASENAME}.npz"
echo "========================================"
