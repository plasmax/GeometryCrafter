# Per-Frame Processing Mode

## Overview

Steps 1 and 2 have been refactored to use **per-frame processing** to minimize VRAM usage. Instead of accumulating all frames in memory, each frame is processed individually and saved immediately to disk.

## What Changed

### Original Approach
- **Step 1**: Load all frames → Process all → Save single `priors.pt` file
- **Step 2**: Load `priors.pt` (all frames) → Process all → Save single `context.pt` file
- **VRAM Issue**: All frame data accumulated in memory

### Per-Frame Approach
- **Step 1**: For each frame → Process → Save `frame_XXXXX_prior.pt` → Free memory
- **Step 2**: For each frame → Load prior → Process → Save 3 files → Free memory
- **VRAM Benefit**: Only one frame in memory at a time

## File Structure

### Step 1 Output (Priors Directory)
```
{TEMP_DIR}/{video_name}_priors/
├── frame_00000_prior.pt
├── frame_00001_prior.pt
├── frame_00002_prior.pt
└── ... (one file per frame)
```

Each `frame_XXXXX_prior.pt` contains:
- `disparity`: [H, W] - Normalized disparity map
- `valid_mask`: [H, W] - Validity mask
- `point_map`: [3, H, W] - 3D point map [x/z, y/z, log(z)]
- `intrinsic_map`: [4, H, W] - Camera intrinsic parameters

### Step 2 Output (Context Directory)
```
{TEMP_DIR}/{video_name}_context/
├── frame_00000_embed.pt           (CLIP embedding)
├── frame_00000_vae_latent.pt      (VAE latent)
├── frame_00000_prior_latent.pt    (Prior latent)
├── frame_00001_embed.pt
├── frame_00001_vae_latent.pt
├── frame_00001_prior_latent.pt
└── ... (3 files per frame)
```

Each frame has 3 files:
- `frame_XXXXX_embed.pt`: CLIP image embedding [1, 1024]
- `frame_XXXXX_vae_latent.pt`: VAE-encoded frame [C, H/8, W/8]
- `frame_XXXXX_prior_latent.pt`: PointMapVAE-encoded prior [C, H/8, W/8]

## Usage

### Step 1: Generate Priors (Per-Frame)
```bash
python bin/step1_priors.py \
    --video_path video.mp4 \
    --output_dir workspace/temp/video_priors \
    --video_info_path workspace/temp/video_info.pt \
    --cache_dir workspace/cache \
    --height 576 \
    --width 1024
```

**Key Changes:**
- `--output_path` → `--output_dir` (now a directory, not a single file)
- Removed `--decode_chunk_size` (processes one frame at a time)

### Step 2: Encode Context (Per-Frame)
```bash
python bin/step2_encode.py \
    --video_path video.mp4 \
    --priors_dir workspace/temp/video_priors \
    --video_info_path workspace/temp/video_info.pt \
    --output_dir workspace/temp/video_context \
    --cache_dir workspace/cache
```

**Key Changes:**
- `--priors_path` → `--priors_dir` (reads per-frame files)
- `--output_path` → `--output_dir` (writes per-frame files)
- Removed `--decode_chunk_size` (processes one frame at a time)

### Orchestrator Script
The `run_geocrafter.sh` script has been updated to use per-frame mode:

```bash
./bin/run_geocrafter.sh --video_path video.mp4
```

**Note**: Steps 3 and 4 still need to be updated to work with per-frame files. The orchestrator currently only runs Steps 1 and 2.

## Memory Benefits

### VRAM Comparison (Example: 100 frames at 576x1024)

#### Original Mode
| Component | VRAM Used |
|-----------|-----------|
| MoGe model | ~4 GB |
| All frame priors (accumulated) | ~2-3 GB |
| **Step 1 Peak** | **~6-7 GB** |
| | |
| VAE + PointMapVAE | ~6 GB |
| All priors loaded | ~2-3 GB |
| All latents (accumulated) | ~3-4 GB |
| **Step 2 Peak** | **~11-13 GB** |

#### Per-Frame Mode
| Component | VRAM Used |
|-----------|-----------|
| MoGe model | ~4 GB |
| Single frame prior | ~30 MB |
| **Step 1 Peak** | **~4.03 GB** |
| | |
| VAE + PointMapVAE | ~6 GB |
| Single frame data | ~30 MB |
| **Step 2 Peak** | **~6.03 GB** |

**VRAM Reduction:**
- Step 1: ~6-7 GB → ~4 GB (40% reduction)
- Step 2: ~11-13 GB → ~6 GB (54% reduction)

## Implementation Details

### Step 1: Two-Pass Processing

**Pass 1: Compute Global Statistics**
- Process all frames through MoGe
- Keep point maps and masks in memory (on CPU)
- Compute global min/max disparity for normalization
- This ensures consistent normalization across all frames

**Pass 2: Normalize and Save Per-Frame**
- For each frame:
  - Normalize using global statistics
  - Resize if needed
  - Save to disk
  - Free memory

**Why Two Passes?**
The disparity normalization requires global statistics (min/max across all frames) to ensure consistency. Without this, each frame would be normalized differently, breaking the temporal coherence.

### Step 2: Three Sequential Encoders

**Encoder 1: CLIP (Per-Frame)**
- Load CLIP encoder
- For each frame:
  - Encode to embedding
  - Save `frame_XXXXX_embed.pt`
  - Free frame data
- Unload CLIP encoder

**Encoder 2: VAE (Per-Frame)**
- Load VAE
- For each frame:
  - Encode to latent
  - Save `frame_XXXXX_vae_latent.pt`
  - Free frame data
- Keep VAE loaded (needed for next step)

**Encoder 3: PointMapVAE (Per-Frame)**
- Load PointMapVAE
- For each frame:
  - Load prior from Step 1
  - Encode using VAE + PointMapVAE
  - Save `frame_XXXXX_prior_latent.pt`
  - Free data
- Unload both VAE and PointMapVAE

## Next Steps (TODO)

### Step 3: Denoising
Needs to be updated to:
- Load per-frame embeddings and latents from context directory
- Process sliding windows by loading only required frames
- Save denoised output (could remain as single file or be per-frame)

### Step 4: Decoding
Needs to be updated to:
- Load per-frame denoised latents (if Step 3 outputs per-frame)
- Decode per-frame or in chunks
- Save final output

## Disk Space Requirements

Per-frame mode uses more disk space but less VRAM:

**For 100 frames at 576x1024:**
- Step 1 output: ~100 files × 1-2 MB = 100-200 MB
- Step 2 output: ~300 files × 500 KB-1 MB = 150-300 MB
- **Total**: ~250-500 MB (vs ~150-250 MB for single-file mode)

Trade-off: ~2x disk space for ~50% less VRAM

## Performance

**Processing Time:**
- Step 1: Similar to original (two-pass overhead is minimal)
- Step 2: Slightly slower due to file I/O overhead (~10-20%)

**Recommendation:**
Use SSD for temp directory to minimize I/O overhead.

## Troubleshooting

### "Out of memory" in Step 1 Pass 1
Pass 1 still accumulates all raw priors on CPU to compute global statistics. If this fails:
1. Reduce video length (`--process_length`)
2. Increase `--downsample_ratio`
3. Use lower resolution (`--height`, `--width`)

### "File not found" errors in Step 2
Step 2 expects per-frame files from Step 1. Ensure:
1. Step 1 completed successfully
2. `--priors_dir` points to Step 1 output directory
3. Frame numbering is consistent (00000, 00001, etc.)

### Slow performance
- Use SSD for `--output_dir`
- Check disk I/O with `iotop` (Linux) or Task Manager (Windows)
- Consider fewer frames for testing

## Migration from Original

If you have existing code using the original single-file mode, note these changes:

**Loading Priors (Old):**
```python
priors = torch.load('priors.pt')
disparity = priors['disparity']  # [T, H, W]
```

**Loading Priors (New):**
```python
disparity_frames = []
for i in range(num_frames):
    prior = torch.load(f'priors/frame_{i:05d}_prior.pt')
    disparity_frames.append(prior['disparity'])  # [H, W]
disparity = torch.stack(disparity_frames)  # [T, H, W]
```

**Or load per-frame as needed:**
```python
# Only load specific frame
prior = torch.load(f'priors/frame_{frame_idx:05d}_prior.pt')
disparity = prior['disparity']  # [H, W]
```

## Summary

✅ **Benefits:**
- 40-50% less VRAM in Steps 1 and 2
- Can process longer videos on limited VRAM
- Better isolation between frames

⚠️ **Trade-offs:**
- More disk space (~2x)
- Slightly slower (~10-20%)
- Steps 3 and 4 need updating

🎯 **Best For:**
- Systems with 16GB or less VRAM
- Very long videos (>200 frames)
- When Step 2 runs out of memory in original mode
