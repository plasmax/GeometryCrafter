# GeometryCrafter Low-Memory Pipeline

This directory contains a memory-optimized inference pipeline for GeometryCrafter, designed to run on systems with limited VRAM (16GB or less).

## Overview

The original `run.py` loads all models simultaneously into VRAM, requiring significant memory. This pipeline splits the inference into 4 isolated steps, loading only one model at a time and caching intermediate results to disk.

### Architecture

```
Video Input
    ↓
[Step 1] MoGe → Geometry Priors → priors.pt
    ↓
[Step 2] Image Encoder + VAE + PointMapVAE → Context → context.pt
    ↓
[Step 3] UNet (VRAM intensive) → Denoised Latents → denoised.pt
    ↓
[Step 4] PointMapVAE Decoder → Final Output.npz
```

### Memory Strategy: "Load-Run-Dump-Kill"

Each step:
1. Loads only the required model(s)
2. Processes the data
3. Saves results to disk as `.pt` files
4. Exits and frees all VRAM

This allows the UNet (Step 3) to use the full 16GB VRAM without competing with other models.

## Files

- `run_geocrafter.sh` - Master orchestrator script
- `step1_priors.py` - Geometry priors generation (MoGe)
- `step2_encode.py` - Context encoding (Image Encoder, VAE, PointMapVAE)
- `step3_denoise.py` - Denoising with UNet (VRAM intensive)
- `step4_decode.py` - Decode to final geometry maps

## Usage

### Basic Usage

```bash
./bin/run_geocrafter.sh --video_path path/to/video.mp4
```

### Full Options

```bash
./bin/run_geocrafter.sh \
    --video_path path/to/video.mp4 \
    --output_dir workspace/output \
    --cache_dir workspace/cache \
    --temp_dir workspace/temp \
    --height 576 \
    --width 1024 \
    --num_inference_steps 5 \
    --window_size 110 \
    --overlap 25 \
    --model_type diff \
    --seed 42
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--video_path` | (required) | Input video file path |
| `--output_dir` | workspace/output | Output directory |
| `--temp_dir` | workspace/temp | Temporary intermediate files |
| `--cache_dir` | workspace/cache | Model cache directory |
| `--height` | (auto) | Target height (must be divisible by 64) |
| `--width` | (auto) | Target width (must be divisible by 64) |
| `--num_inference_steps` | 5 | Number of denoising steps |
| `--guidance_scale` | 1.0 | Classifier-free guidance scale |
| `--window_size` | 110 | Sliding window size for long videos |
| `--overlap` | 25 | Overlap between windows |
| `--model_type` | diff | Model type: 'diff' or 'determ' |
| `--seed` | 42 | Random seed for reproducibility |

## Step-by-Step Breakdown

### Step 1: Geometry Priors (MoGe)

**Models loaded:** MoGe (Monocular Geometry Estimator)

**Operations:**
- Reads video frames
- Generates per-frame disparity, validity masks, point maps, and intrinsic maps
- Normalizes and processes geometry priors

**Output:** `priors.pt` containing:
- `disparity` - Normalized disparity maps (T,H,W)
- `valid_mask` - Validity masks (T,H,W)
- `point_map` - 3D point maps in [x/z, y/z, log(z)] format (T,3,H,W)
- `intrinsic_map` - Camera intrinsic maps (T,4,H,W)

**VRAM Impact:** Moderate (MoGe only)

### Step 2: Context Encoding

**Models loaded sequentially (never simultaneously):**
1. CLIP Image Encoder
2. VAE (AutoencoderKLTemporalDecoder)
3. PointMapVAE

**Operations:**
1. Load Image Encoder → Generate video embeddings → Unload
2. Load VAE → Encode video frames to latents → Keep loaded
3. Load PointMapVAE → Encode geometry priors to latents → Unload both

**Output:** `context.pt` containing:
- `video_embeddings` - CLIP embeddings (1,T,1024)
- `video_latents` - VAE-encoded video (1,T,C,H/8,W/8)
- `prior_latents` - Encoded geometry priors (1,T,C,H/8,W/8)

**VRAM Impact:** Moderate (sequential loading)

### Step 3: Denoising (UNet)

**Models loaded:** UNetSpatioTemporalConditionModelVid2vid + Scheduler

**Operations:**
- Loads context from Step 2
- Performs iterative denoising using sliding window approach
- Applies classifier-free guidance if `guidance_scale > 1.0`

**Output:** `denoised.pt` containing:
- `latents` - Denoised latent representations (T,C,H/8,W/8)

**VRAM Impact:** **HIGH** - This is the most VRAM-intensive step. By isolating it, the UNet gets full access to 16GB.

### Step 4: Decoding

**Models loaded:** PointMapVAE

**Operations:**
- Decodes latents back to geometry maps
- Reconstructs 3D point clouds
- Applies projection constraints

**Output:** `{video_name}.npz` containing:
- `point_map` - 3D point maps (T,H,W,3) as float16
- `mask` - Validity masks (T,H,W) as bool

**VRAM Impact:** Low

## Intermediate Files

The orchestrator creates temporary files in `--temp_dir`:
- `{video}_priors.pt` - Geometry priors from Step 1
- `{video}_context.pt` - Encoded context from Step 2
- `{video}_denoised.pt` - Denoised latents from Step 3
- `{video}_video_info.pt` - Metadata (dimensions, etc.)

These files are automatically cleaned up after successful completion.

## Comparison with Original

| Aspect | Original `run.py` | Low-Memory Pipeline |
|--------|-------------------|---------------------|
| VRAM Usage | All models loaded simultaneously | One model at a time |
| Peak VRAM | ~24-32GB | ~14-16GB |
| Speed | Faster (no disk I/O) | Slower (disk caching) |
| Flexibility | Single script | Modular steps |
| Resume Support | No | Can resume from any step |

## Troubleshooting

### Out of Memory Errors

If you still encounter OOM errors:

1. Reduce `--window_size` (try 80 or 60)
2. Reduce `--decode_chunk_size` (try 4 or 2)
3. Use `--downsample_ratio 2.0` to process at half resolution
4. Set `--guidance_scale 1.0` to disable CFG (saves ~30% VRAM in Step 3)

### Manual Step Execution

You can run steps individually for debugging:

```bash
# Step 1
python bin/step1_priors.py \
    --video_path video.mp4 \
    --output_path temp/priors.pt \
    --video_info_path temp/info.pt

# Step 2
python bin/step2_encode.py \
    --video_path video.mp4 \
    --priors_path temp/priors.pt \
    --video_info_path temp/info.pt \
    --output_path temp/context.pt

# Step 3
python bin/step3_denoise.py \
    --context_path temp/context.pt \
    --video_info_path temp/info.pt \
    --output_path temp/denoised.pt

# Step 4
python bin/step4_decode.py \
    --denoised_path temp/denoised.pt \
    --video_info_path temp/info.pt \
    --output_dir output \
    --video_basename video
```

## Performance Tips

1. **SSD Storage**: Use an SSD for `--temp_dir` to minimize I/O overhead
2. **Smaller Windows**: Reduce `--window_size` for very long videos
3. **Fewer Steps**: Use `--num_inference_steps 3` for faster (but lower quality) results
4. **Batch Size**: This pipeline always uses batch_size=1 for memory efficiency

## License

Same as parent GeometryCrafter project.
