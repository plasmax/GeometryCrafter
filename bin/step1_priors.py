#!/usr/bin/env python3
"""
Step 1: Geometry Priors Generation
Loads MoGe model, processes video frames, generates disparity/masks/point maps.
Saves results to disk and exits to free VRAM.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from third_party import MoGe
from geometrycrafter.diff_ppl import normalize_point_map, point_map_xy2intrinsic_map, robust_min_max


def produce_priors(prior_model, frames, chunk_size=8, device='cuda'):
    """Generate geometry priors from video frames."""
    T, _, H, W = frames.shape

    pred_point_maps = []
    pred_masks = []

    print(f"Processing {T} frames in chunks of {chunk_size}...")
    for i in range(0, T, chunk_size):
        end_idx = min(i + chunk_size, T)
        chunk = frames[i:end_idx].to(device)

        pred_p, pred_m = prior_model.forward_image(chunk)

        # Move back to CPU to save VRAM
        pred_point_maps.append(pred_p.cpu())
        pred_masks.append(pred_m.cpu())

        print(f"  Processed frames {i+1}-{end_idx}/{T}")

    pred_point_maps = torch.cat(pred_point_maps, dim=0)
    pred_masks = torch.cat(pred_masks, dim=0)

    # Convert masks to [-1, 1] range
    pred_masks = pred_masks.float() * 2 - 1

    # Normalize point maps (T,H,W,3) with valid mask (T,H,W)
    pred_point_maps = normalize_point_map(pred_point_maps, pred_masks > 0)

    # Compute disparity from depth (1/z)
    pred_disps = 1.0 / pred_point_maps[..., 2].clamp_min(1e-3)
    pred_disps = pred_disps * (pred_masks > 0)
    min_disparity, max_disparity = robust_min_max(pred_disps)
    pred_disps = ((pred_disps - min_disparity) / (max_disparity - min_disparity + 1e-4)).clamp(0, 1)
    pred_disps = pred_disps * 2 - 1  # Normalize to [-1, 1]

    # Convert point map to [x/z, y/z, log(z)]
    pred_point_maps[..., :2] = pred_point_maps[..., :2] / (pred_point_maps[..., 2:3] + 1e-7)
    pred_point_maps[..., 2] = torch.log(pred_point_maps[..., 2] + 1e-7) * (pred_masks > 0)

    # Compute intrinsic maps (T,4,H,W)
    pred_intr_maps = point_map_xy2intrinsic_map(pred_point_maps[..., :2]).permute(0, 3, 1, 2)
    pred_point_maps = pred_point_maps.permute(0, 3, 1, 2)

    return pred_disps, pred_masks, pred_point_maps, pred_intr_maps


def main():
    parser = argparse.ArgumentParser(description='Step 1: Generate Geometry Priors')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_path', type=str, required=True, help='Output .pt file for priors')
    parser.add_argument('--video_info_path', type=str, required=True, help='Output .pt file for video info')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')
    parser.add_argument('--height', type=str, default='', help='Target height (empty = auto)')
    parser.add_argument('--width', type=str, default='', help='Target width (empty = auto)')
    parser.add_argument('--downsample_ratio', type=float, default=1.0, help='Downsample ratio')
    parser.add_argument('--decode_chunk_size', type=int, default=8, help='Chunk size for processing')
    parser.add_argument('--process_length', type=int, default=-1, help='Number of frames to process (-1 = all)')
    parser.add_argument('--process_stride', type=int, default=1, help='Frame stride')

    args = parser.parse_args()

    print("="*60)
    print("Step 1: Geometry Priors Generation (MoGe)")
    print("="*60)

    # Load video
    print(f"Loading video: {args.video_path}")
    vid = VideoReader(args.video_path, ctx=cpu(0))
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    # Determine target dimensions
    if args.height and args.width:
        height = int(args.height)
        width = int(args.width)
    else:
        height = original_height
        width = original_width

    assert height % 64 == 0, f"Height must be divisible by 64, got {height}"
    assert width % 64 == 0, f"Width must be divisible by 64, got {width}"

    # Load frames
    frames_idx = list(range(0, len(vid), args.process_stride))
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32) / 255.0

    if args.process_length > 0:
        process_length = min(args.process_length, len(frames))
        frames = frames[:process_length]
    else:
        process_length = len(frames)

    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
    print(f"Loaded {process_length} frames at {original_height}x{original_width}")

    # Apply downsampling if needed
    if args.downsample_ratio > 1.0:
        new_h = round(frames_tensor.shape[-2] / args.downsample_ratio)
        new_w = round(frames_tensor.shape[-1] / args.downsample_ratio)
        frames_tensor = F.interpolate(
            frames_tensor, (new_h, new_w),
            mode='bicubic', antialias=True
        ).clamp(0, 1)
        print(f"Downsampled to {new_h}x{new_w} (ratio={args.downsample_ratio})")

    # Load MoGe model
    print(f"Loading MoGe model (cache: {args.cache_dir})...")
    prior_model = MoGe(cache_dir=args.cache_dir).requires_grad_(False).to('cuda', dtype=torch.float32)
    print("MoGe model loaded.")

    # Generate priors
    print("Generating geometry priors...")
    with torch.inference_mode():
        pred_disparity, pred_valid_mask, pred_point_map, pred_intrinsic_map = produce_priors(
            prior_model,
            frames_tensor,
            chunk_size=args.decode_chunk_size,
            device='cuda'
        )

    # Resize if needed
    need_resize = (original_height != height or original_width != width)
    if need_resize:
        print(f"Resizing priors to {height}x{width}...")
        pred_disparity = F.interpolate(
            pred_disparity.unsqueeze(1), (height, width),
            mode='bilinear', align_corners=False
        ).squeeze(1)
        pred_valid_mask = F.interpolate(
            pred_valid_mask.unsqueeze(1), (height, width),
            mode='bilinear', align_corners=False
        ).squeeze(1)

        # Transform log-depth to depth domain for interpolation
        pred_point_map = torch.cat([
            F.interpolate(pred_point_map[:, 0:2], (height, width), mode='bilinear', align_corners=False),
            F.interpolate(
                pred_point_map[:, 2:3].clamp_max(10).exp(), (height, width),
                mode='bilinear', align_corners=False
            ).log()
        ], dim=1)
        pred_intrinsic_map = F.interpolate(
            pred_intrinsic_map, (height, width),
            mode='bilinear', align_corners=False
        )

    # Save priors to disk
    print(f"Saving priors to {args.output_path}...")
    torch.save({
        'disparity': pred_disparity,
        'valid_mask': pred_valid_mask,
        'point_map': pred_point_map,
        'intrinsic_map': pred_intrinsic_map,
    }, args.output_path)

    # Save video info (needed by subsequent steps)
    print(f"Saving video info to {args.video_info_path}...")
    torch.save({
        'num_frames': process_length,
        'height': height,
        'width': width,
        'original_height': original_height,
        'original_width': original_width,
        'need_resize': need_resize,
        'downsample_ratio': args.downsample_ratio,
    }, args.video_info_path)

    print("Step 1 complete!")
    print(f"  Priors shape: disparity={list(pred_disparity.shape)}, "
          f"point_map={list(pred_point_map.shape)}")


if __name__ == '__main__':
    main()
