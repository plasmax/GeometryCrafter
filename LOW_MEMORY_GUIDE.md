# GeometryCrafter: Low-Memory Inference Guide

## Quick Start

If you have limited VRAM (16GB or less), use the low-memory pipeline instead of `run.py`:

```bash
# Original (requires ~24-32GB VRAM)
python run.py --video_path video.mp4

# Low-memory pipeline (requires ~14-16GB VRAM)
./bin/run_geocrafter.sh --video_path video.mp4
```

## What's Different?

The low-memory pipeline splits inference into 4 isolated steps, loading only one model at a time:

1. **Step 1: MoGe** - Generate geometry priors
2. **Step 2: Encoders** - Encode context (CLIP + VAE + PointMapVAE)
3. **Step 3: UNet** - Denoise latents (VRAM intensive)
4. **Step 4: Decoder** - Decode to final output

See [bin/README.md](bin/README.md) for full documentation.

## VRAM Comparison

| Model/Step | Original | Low-Memory |
|------------|----------|------------|
| MoGe | ✓ Loaded | Step 1 only |
| Image Encoder | ✓ Loaded | Step 2 only |
| VAE | ✓ Loaded | Steps 2 only |
| PointMapVAE | ✓ Loaded | Steps 2 & 4 only |
| UNet | ✓ Loaded | Step 3 only |
| **Peak VRAM** | **~24-32GB** | **~14-16GB** |

## Trade-offs

**Advantages:**
- Runs on 16GB VRAM systems
- Modular - can resume from any step
- Easier to debug individual stages

**Disadvantages:**
- Slower due to disk I/O
- Requires temporary storage for intermediate files (~2-4GB per video)

## When to Use Which

- **Use `run.py`** if you have 24GB+ VRAM and want maximum speed
- **Use `bin/run_geocrafter.sh`** if you have 16GB or less VRAM
