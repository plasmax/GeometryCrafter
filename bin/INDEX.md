# Low-Memory Pipeline - File Index

## Quick Start

**New to the low-memory pipeline?** Start here:
1. Read [../LOW_MEMORY_GUIDE.md](../LOW_MEMORY_GUIDE.md) - 2 min overview
2. Read [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - Essential commands
3. Run `./bin/test_pipeline.sh` to validate your setup
4. Run `./bin/run_geocrafter.sh --video_path your_video.mp4`

## File Organization

### Core Pipeline (Executable Scripts)

| File | Purpose | Lines | Run Order |
|------|---------|-------|-----------|
| [run_geocrafter.sh](run_geocrafter.sh) | Master orchestrator | ~200 | First |
| [step1_priors.py](step1_priors.py) | MoGe geometry priors | ~180 | 1/4 |
| [step2_encode.py](step2_encode.py) | Context encoding | ~240 | 2/4 |
| [step3_denoise.py](step3_denoise.py) | UNet denoising | ~230 | 3/4 |
| [step4_decode.py](step4_decode.py) | Decode to geometry | ~200 | 4/4 |

### Utilities

| File | Purpose | Usage |
|------|---------|-------|
| [test_pipeline.sh](test_pipeline.sh) | Validate setup | `./bin/test_pipeline.sh` |
| [cleanup_temp.sh](cleanup_temp.sh) | Clean temp files | `./bin/cleanup_temp.sh` |

### Documentation

| File | Type | Audience | Read Time |
|------|------|----------|-----------|
| [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) | Quick ref | All users | 2 min |
| [README.md](README.md) | Full guide | All users | 10 min |
| [EXAMPLES.md](EXAMPLES.md) | Practical examples | Users | 5 min |
| [PIPELINE_DIAGRAM.txt](PIPELINE_DIAGRAM.txt) | Visual docs | Technical | 5 min |
| [../LOW_MEMORY_GUIDE.md](../LOW_MEMORY_GUIDE.md) | Overview | New users | 2 min |
| [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) | Technical deep-dive | Developers | 15 min |
| [../CHANGELOG_LOW_MEMORY.md](../CHANGELOG_LOW_MEMORY.md) | Implementation log | All | 5 min |

## Documentation Hierarchy

```
GeometryCrafter/
│
├─ LOW_MEMORY_GUIDE.md          ⭐ START HERE (overview)
│
├─ bin/
│  ├─ QUICK_REFERENCE.txt       ⭐ QUICK COMMANDS
│  ├─ README.md                 📖 Complete documentation
│  ├─ EXAMPLES.md               💡 Practical examples
│  ├─ PIPELINE_DIAGRAM.txt      📊 Visual architecture
│  └─ INDEX.md                  📑 This file
│
├─ IMPLEMENTATION_SUMMARY.md    🔧 Technical details
└─ CHANGELOG_LOW_MEMORY.md      📝 Implementation log
```

## Recommended Reading Order

### For Users (Just want to run it)
1. [../LOW_MEMORY_GUIDE.md](../LOW_MEMORY_GUIDE.md) - What is this?
2. [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - How to run it?
3. [EXAMPLES.md](EXAMPLES.md) - Common use cases
4. [README.md](README.md) - Full documentation (as needed)

### For Developers (Want to understand/modify)
1. [../LOW_MEMORY_GUIDE.md](../LOW_MEMORY_GUIDE.md) - Overview
2. [PIPELINE_DIAGRAM.txt](PIPELINE_DIAGRAM.txt) - Architecture
3. [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md) - Design decisions
4. [README.md](README.md) - Step-by-step details
5. Source code: [step1_priors.py](step1_priors.py), [step2_encode.py](step2_encode.py), etc.

### For Troubleshooting
1. [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt) - Common issues
2. [README.md](README.md) - Troubleshooting section
3. [EXAMPLES.md](EXAMPLES.md) - Debugging examples

## File Contents Summary

### [run_geocrafter.sh](run_geocrafter.sh)
Bash orchestrator that:
- Parses command-line arguments
- Creates temporary directories
- Runs all 4 steps sequentially
- Cleans up intermediate files
- Handles errors and interruptions

### [step1_priors.py](step1_priors.py)
MoGe-based geometry prior generation:
- Loads video frames
- Runs MoGe monocular depth estimation
- Generates disparity, masks, point maps, intrinsics
- Saves to `priors.pt`
- VRAM: ~4-6 GB

### [step2_encode.py](step2_encode.py)
Sequential encoding:
- CLIP: Video → embeddings
- VAE: Video → latents
- PointMapVAE: Priors → latents
- Saves to `context.pt`
- VRAM: ~6-8 GB (sequential loading)

### [step3_denoise.py](step3_denoise.py)
UNet denoising (VRAM-intensive):
- Loads UNet + scheduler only
- Sliding window denoising
- Optional classifier-free guidance
- Saves to `denoised.pt`
- VRAM: ~12-16 GB (peak)

### [step4_decode.py](step4_decode.py)
PointMapVAE decoding:
- Decodes latents to geometry maps
- Reconstructs 3D point clouds
- Applies projection constraints
- Saves to `{video}.npz`
- VRAM: ~4-6 GB

### [test_pipeline.sh](test_pipeline.sh)
Validation script that checks:
- All scripts present
- Python syntax valid
- Dependencies installed
- CUDA available
- VRAM sufficient

### [cleanup_temp.sh](cleanup_temp.sh)
Cleanup utility:
- Lists temporary files
- Shows sizes
- Interactive deletion

### [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)
One-page reference with:
- Essential commands
- Common scenarios
- Parameter presets
- Troubleshooting checklist

### [README.md](README.md)
Complete documentation:
- Architecture overview
- Usage instructions
- All parameters explained
- Step-by-step breakdown
- Troubleshooting guide
- Performance tips

### [EXAMPLES.md](EXAMPLES.md)
20+ practical examples:
- Basic usage
- Memory optimization
- Quality tuning
- Batch processing
- Debugging techniques

### [PIPELINE_DIAGRAM.txt](PIPELINE_DIAGRAM.txt)
ASCII visualizations:
- Architecture diagrams
- Data flow charts
- Memory profiles
- Comparison tables

## Quick Command Reference

```bash
# Validate setup
./bin/test_pipeline.sh

# Basic inference
./bin/run_geocrafter.sh --video_path video.mp4

# Low VRAM mode
./bin/run_geocrafter.sh --video_path video.mp4 \
    --window_size 60 --decode_chunk_size 4 --guidance_scale 1.0

# High quality
./bin/run_geocrafter.sh --video_path video.mp4 \
    --num_inference_steps 10 --guidance_scale 2.0

# Clean up
./bin/cleanup_temp.sh
```

## Help & Support

- **Getting Started**: See [QUICK_REFERENCE.txt](QUICK_REFERENCE.txt)
- **Full Documentation**: See [README.md](README.md)
- **Examples**: See [EXAMPLES.md](EXAMPLES.md)
- **Technical Details**: See [../IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)
- **Troubleshooting**: All docs have troubleshooting sections

## File Statistics

| Category | Count | Total Lines |
|----------|-------|-------------|
| Core Scripts (Python) | 4 | ~850 |
| Orchestrator (Bash) | 1 | ~200 |
| Utilities (Bash) | 2 | ~100 |
| Documentation (MD/TXT) | 7 | ~1,500+ |
| **Total** | **14** | **~2,650+** |

## Version

**Pipeline Version**: 1.0.0
**Implementation Date**: January 2026
**Strategy**: Load-Run-Dump-Kill (Sequential Model Loading)
**Target**: 16GB VRAM systems
