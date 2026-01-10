# Usage Examples

## Basic Examples

### 1. Simple Inference (Auto-detect resolution)

```bash
./bin/run_geocrafter.sh --video_path videos/sample.mp4
```

Output: `workspace/output/sample.npz`

### 2. Specify Resolution

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --height 576 \
    --width 1024
```

Resolution must be divisible by 64.

### 3. Process Only First 100 Frames

```bash
./bin/run_geocrafter.sh \
    --video_path videos/long_video.mp4 \
    --process_length 100
```

### 4. High Quality (More Denoising Steps)

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --num_inference_steps 10 \
    --guidance_scale 2.0
```

More steps = better quality but slower. CFG guidance > 1.0 uses more VRAM.

## Memory Optimization Examples

### 5. Minimal VRAM Usage

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --window_size 60 \
    --overlap 15 \
    --decode_chunk_size 4 \
    --guidance_scale 1.0
```

This configuration minimizes VRAM at the cost of speed.

### 6. Downsample for Very High-Res Videos

```bash
./bin/run_geocrafter.sh \
    --video_path videos/4k_video.mp4 \
    --downsample_ratio 2.0 \
    --height 512 \
    --width 896
```

Process at half resolution, then upsample results.

### 7. Fast Preview (Lower Quality)

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --num_inference_steps 3 \
    --process_stride 2
```

Use fewer denoising steps and skip every other frame for quick results.

## Advanced Examples

### 8. Custom Cache and Output Directories

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --output_dir results/experiment1 \
    --cache_dir /mnt/ssd/models \
    --temp_dir /mnt/ssd/temp
```

Use SSD for temp files to improve performance.

### 9. Deterministic Model

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --model_type determ
```

Use deterministic model instead of diffusion model.

### 10. Long Video Processing

```bash
./bin/run_geocrafter.sh \
    --video_path videos/long_video.mp4 \
    --window_size 110 \
    --overlap 25
```

The pipeline automatically handles long videos with sliding windows.

### 11. Reproducible Results

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --seed 12345
```

Use a fixed seed for reproducible outputs.

### 12. Disable Projection Constraints

```bash
./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --no_force_projection \
    --no_force_fixed_focal
```

Allow variable focal length and no projection enforcement.

## Batch Processing Examples

### 13. Process Multiple Videos

```bash
#!/bin/bash
for video in videos/*.mp4; do
    echo "Processing $video..."
    ./bin/run_geocrafter.sh --video_path "$video"
done
```

### 14. Parallel Processing (Multiple GPUs)

```bash
#!/bin/bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 ./bin/run_geocrafter.sh --video_path video1.mp4 &

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 ./bin/run_geocrafter.sh --video_path video2.mp4 &

wait
```

### 15. Resume from Specific Step

If a step fails, you can resume manually:

```bash
# Pipeline failed at Step 3
# Manually run Step 3 and Step 4:

python bin/step3_denoise.py \
    --context_path workspace/temp/video_context.pt \
    --video_info_path workspace/temp/video_video_info.pt \
    --output_path workspace/temp/video_denoised.pt \
    --num_inference_steps 5

python bin/step4_decode.py \
    --denoised_path workspace/temp/video_denoised.pt \
    --video_info_path workspace/temp/video_video_info.pt \
    --output_dir workspace/output \
    --video_basename video
```

## Debugging Examples

### 16. Test Pipeline Setup

```bash
./bin/test_pipeline.sh
```

### 17. Check Intermediate Outputs

```bash
# After running Step 1
python -c "
import torch
priors = torch.load('workspace/temp/video_priors.pt')
print('Disparity shape:', priors['disparity'].shape)
print('Point map shape:', priors['point_map'].shape)
"
```

### 18. Monitor VRAM Usage

```bash
# In another terminal while pipeline runs
watch -n 1 nvidia-smi
```

### 19. Clean Up Temp Files

```bash
# Interactive cleanup
./bin/cleanup_temp.sh

# Force cleanup
rm -rf workspace/temp/*
```

## Performance Comparison

### 20. Compare Original vs Low-Memory

```bash
# Original (if you have 24GB+ VRAM)
time python run.py \
    --video_path videos/sample.mp4 \
    --save_folder results/original

# Low-memory pipeline
time ./bin/run_geocrafter.sh \
    --video_path videos/sample.mp4 \
    --output_dir results/lowmem

# Compare results
python -c "
import numpy as np
orig = np.load('results/original/sample.npz')
lowm = np.load('results/lowmem/sample.npz')
print('Point map difference:', np.abs(orig['point_map'] - lowm['point_map']).mean())
"
```

Results should be nearly identical (small differences due to floating-point precision).

## Common Use Cases

### Scientific Research

```bash
./bin/run_geocrafter.sh \
    --video_path data/experiment_001.mp4 \
    --num_inference_steps 10 \
    --seed 42 \
    --output_dir results/exp001
```

### Quick Preview/Demo

```bash
./bin/run_geocrafter.sh \
    --video_path demo.mp4 \
    --num_inference_steps 3 \
    --process_length 50 \
    --guidance_scale 1.0
```

### Production (Best Quality)

```bash
./bin/run_geocrafter.sh \
    --video_path production.mp4 \
    --num_inference_steps 10 \
    --guidance_scale 2.0 \
    --window_size 110 \
    --decode_chunk_size 8
```

## Tips

1. **Start Small:** Test with short videos first to validate settings
2. **Monitor VRAM:** Use `nvidia-smi` to ensure you're not hitting limits
3. **Use SSD:** Place `--temp_dir` on SSD for better performance
4. **Experiment:** Try different `num_inference_steps` values (3, 5, 10)
5. **Resolution:** Lower resolution = faster processing + less VRAM
