#!/bin/bash

# Commands for running each step of the GeometryCrafter pipeline manually.

VIDEO_PATH="examples/video4.mp4"
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
ATTENTION_MODE="auto"
CPU_OFFLOAD=false
RESIDUAL_OFFLOAD="none"
RESIDUAL_CACHE_DIR="$TEMP_DIR/residuals"
ENABLE_UNET_CHECKPOINTING=false
FORCE_PROJECTION=true
FORCE_FIXED_FOCAL=true
USE_EXTRACT_INTERP=false
ENABLE_VAE_OFFLOADING=true
ENABLE_INFERENCE_CHECKPOINTING=true

VIDEO_BASENAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
PRIORS_DIR="$TEMP_DIR/${VIDEO_BASENAME}_priors"
CONTEXT_DIR="$TEMP_DIR/${VIDEO_BASENAME}_context"
DENOISED_FILE="$TEMP_DIR/${VIDEO_BASENAME}_denoised.pt"
VIDEO_INFO_FILE="$TEMP_DIR/${VIDEO_BASENAME}_video_info.pt"

# # Step 1: Generate Geometry Priors (Per-Frame)
# python bin/step1_priors.py \
#   --video_path "$VIDEO_PATH" \
#   --output_dir "$PRIORS_DIR" \
#   --video_info_path "$VIDEO_INFO_FILE" \
#   --cache_dir "$CACHE_DIR" \
#   --height "$HEIGHT" \
#   --width "$WIDTH" \
#   --downsample_ratio "$DOWNSAMPLE_RATIO" \
#   --process_length "$PROCESS_LENGTH" \
#   --process_stride "$PROCESS_STRIDE"

# # Step 2: Encode Context (Per-Frame)
# python bin/step2_encode.py \
#   --video_path "$VIDEO_PATH" \
#   --priors_dir "$PRIORS_DIR" \
#   --video_info_path "$VIDEO_INFO_FILE" \
#   --output_dir "$CONTEXT_DIR" \
#   --cache_dir "$CACHE_DIR"

# # Step 3: Denoising (UNet) - Per-Frame Context Loading
# python bin/step3_denoise.py \
#   --context_dir "$CONTEXT_DIR" \
#   --video_info_path "$VIDEO_INFO_FILE" \
#   --output_path "$DENOISED_FILE" \
#   --cache_dir "$CACHE_DIR" \
#   --num_inference_steps "$NUM_INFERENCE_STEPS" \
#   --guidance_scale "$GUIDANCE_SCALE" \
#   --window_size "$WINDOW_SIZE" \
#   --overlap "$OVERLAP" \
#   --seed "$SEED" \
#   --model_type "$MODEL_TYPE" \
#   --attention_mode "$ATTENTION_MODE" \
#   --residual_offload "$RESIDUAL_OFFLOAD" \
#   --residual_cache_dir "$RESIDUAL_CACHE_DIR" \
#   $( [ "$CPU_OFFLOAD" = "true" ] && echo "--cpu_offload" ) \
#   $( [ "$ENABLE_UNET_CHECKPOINTING" = "true" ] && echo "--enable_unet_checkpointing" )

# Step 4: Decode to Geometry Maps
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

# # Step 5: Convert to MP4
# OUTPUT_NPZ="$OUTPUT_DIR/${VIDEO_BASENAME}.npz"
# OUTPUT_MP4="$OUTPUT_DIR/${VIDEO_BASENAME}.mp4"

# python bin/npz_to_mp4.py \
#   --npz_path "$OUTPUT_NPZ" \
#   --output_path "$OUTPUT_MP4" \
#   --fps 30
