# Debug Mode

## Overview

The orchestrator script supports a debug mode that pauses before cleanup when an error occurs, allowing you to inspect intermediate files.

## Usage

### Enable Debug Mode

```bash
DEBUG_MODE=true ./bin/run_geocrafter.sh --video_path video.mp4
```

### What Happens

When an error occurs:
1. Pipeline stops at the failing step
2. **Does NOT immediately cleanup** intermediate files
3. Displays message:
   ```
   ========================================
   ERROR OCCURRED - Debug Mode Active
   ========================================
   Intermediate files preserved in: workspace/temp
   Press ENTER to cleanup and exit, or Ctrl+C to keep files...
   ```
4. Waits for you to press ENTER
5. You can inspect files in another terminal before pressing ENTER
6. Pressing Ctrl+C keeps the files and exits without cleanup

### Inspecting Intermediate Files

While paused, in another terminal:

```bash
# List all intermediate files
ls -lh workspace/temp/

# Check a specific prior file
python -c "
import torch
data = torch.load('workspace/temp/video_priors/frame_00000_prior.pt')
for k, v in data.items():
    print(f'{k}: {v.shape} {v.dtype}')
"

# Check a context file
python -c "
import torch
embed = torch.load('workspace/temp/video_context/frame_00000_embed.pt')
print(f'Embedding: {embed.shape} {embed.dtype}')
"

# Check video info
python -c "
import torch
info = torch.load('workspace/temp/video_video_info.pt')
print(info)
"
```

### Normal Mode (No Debug)

```bash
# Without DEBUG_MODE or DEBUG_MODE=false
./bin/run_geocrafter.sh --video_path video.mp4
```

Automatically cleans up on error (default behavior).

## Use Cases

### 1. Investigating Errors

When you get an error like:
```
RuntimeError: Tensors must have same number of dimensions: got 5 and 6
```

Enable debug mode to:
- Check tensor shapes in intermediate files
- Verify file contents
- Diagnose dimension mismatches

### 2. Verifying File Sizes

```bash
DEBUG_MODE=true ./bin/run_geocrafter.sh --video_path video.mp4
# Let it run Step 1, then Ctrl+C
# Check file sizes without running full pipeline
du -h workspace/temp/video_priors/
```

### 3. Resuming Failed Runs

If Step 3 fails but Steps 1-2 completed:
```bash
# 1. Run with debug mode
DEBUG_MODE=true ./bin/run_geocrafter.sh --video_path video.mp4
# (fails at Step 3)

# 2. Press Ctrl+C to keep files

# 3. Fix the issue, then manually run remaining steps
python bin/step3_denoise.py \
    --context_dir workspace/temp/video_context \
    --video_info_path workspace/temp/video_video_info.pt \
    --output_path workspace/temp/video_denoised.pt \
    ...

python bin/step4_decode.py \
    --denoised_path workspace/temp/video_denoised.pt \
    ...
```

## Tips

1. **Always inspect before cleanup**: When debug mode pauses, take your time to inspect files

2. **Keep problematic files**: Press Ctrl+C instead of ENTER to preserve files for later analysis

3. **Combine with verbose logging**: Add `set -x` at the top of the script for detailed execution trace

4. **Check disk space**: Before keeping large temp files, ensure you have enough space

## Example Session

```bash
$ DEBUG_MODE=true ./bin/run_geocrafter.sh --video_path video.mp4
...
==> Step 3/4: Denoising Latents (UNet - VRAM intensive)
...
RuntimeError: Tensors must have same number of dimensions: got 5 and 6

========================================
ERROR OCCURRED - Debug Mode Active
========================================
Intermediate files preserved in: workspace/temp
Press ENTER to cleanup and exit, or Ctrl+C to keep files...
```

In another terminal:
```bash
$ ls workspace/temp/
video_priors/  video_context/  video_video_info.pt

$ ls workspace/temp/video_context/ | wc -l
330  # 110 frames × 3 files per frame

$ python -c "import torch; print(torch.load('workspace/temp/video_context/frame_00000_vae_latent.pt').shape)"
torch.Size([4, 72, 128])  # [C, H, W] - correct!
```

Back to first terminal:
```bash
Press ENTER to cleanup and exit, or Ctrl+C to keep files...
^C  # Keep files for further debugging
```

## Environment Variable

The debug mode is controlled by the `DEBUG_MODE` environment variable:
- `DEBUG_MODE=true` - Enable debug mode
- `DEBUG_MODE=false` or unset - Normal mode (default)

Can also export it for multiple runs:
```bash
export DEBUG_MODE=true
./bin/run_geocrafter.sh --video_path video1.mp4
./bin/run_geocrafter.sh --video_path video2.mp4
unset DEBUG_MODE
```
