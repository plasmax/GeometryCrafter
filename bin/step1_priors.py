#!/usr/bin/env python3
"""
Step 1: Geometry Priors Generation (Per-Frame)
Loads MoGe model, processes video frames one at a time.
Saves each frame's priors separately to minimize VRAM usage.
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


def process_frame_priors(prior_model, frame, device='cuda'):
    """Generate geometry priors for a single frame."""
    # frame: [1, C, H, W]
    with torch.inference_mode():
        pred_p, pred_m = prior_model.forward_image(frame.to(device))

    # Move back to CPU immediately
    pred_p = pred_p.cpu()
    pred_m = pred_m.cpu()

    return pred_p, pred_m


def postprocess_priors(pred_point_maps, pred_masks):
    """
    Postprocess accumulated priors from all frames.
    This computes global statistics needed for normalization.
    """
    # Convert masks to [-1, 1] range
    pred_masks = pred_masks.float() * 2 - 1

    # Normalize point maps (T,H,W,3) with valid mask (T,H,W)
    pred_point_maps = normalize_point_map(pred_point_maps, pred_masks > 0)

    # Compute disparity from depth (1/z)
    pred_disps = 1.0 / pred_point_maps[..., 2].clamp_min(1e-3)
    pred_disps = pred_disps * (pred_masks > 0)

    # Compute global min/max for normalization
    min_disparity, max_disparity = robust_min_max(pred_disps)

    return min_disparity, max_disparity


def normalize_frame(pred_p, pred_m, min_disparity, max_disparity):
    """Normalize a single frame's priors using global statistics."""
    # pred_p: [1, H, W, 3], pred_m: [1, H, W]

    # Convert mask to [-1, 1]
    pred_m = pred_m.float() * 2 - 1

    # Normalize point map (already done in postprocess, but redo per-frame)
    valid_mask = pred_m > 0

    # Compute disparity
    pred_disp = 1.0 / pred_p[..., 2].clamp_min(1e-3)
    pred_disp = pred_disp * valid_mask

    # Normalize disparity with global stats
    pred_disp = ((pred_disp - min_disparity) / (max_disparity - min_disparity + 1e-4)).clamp(0, 1)
    pred_disp = pred_disp * 2 - 1  # To [-1, 1]

    # Convert point map to [x/z, y/z, log(z)]
    pred_p[..., :2] = pred_p[..., :2] / (pred_p[..., 2:3] + 1e-7)
    pred_p[..., 2] = torch.log(pred_p[..., 2] + 1e-7) * valid_mask

    # Compute intrinsic map
    pred_intr = point_map_xy2intrinsic_map(pred_p[..., :2]).permute(0, 3, 1, 2)
    pred_p = pred_p.permute(0, 3, 1, 2)

    return pred_disp.squeeze(0), pred_m.squeeze(0), pred_p.squeeze(0), pred_intr.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description='Step 1: Generate Geometry Priors (Per-Frame)')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for per-frame .pt files')
    parser.add_argument('--video_info_path', type=str, required=True, help='Output .pt file for video info')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')
    parser.add_argument('--height', type=str, default='', help='Target height (empty = auto)')
    parser.add_argument('--width', type=str, default='', help='Target width (empty = auto)')
    parser.add_argument('--downsample_ratio', type=float, default=1.0, help='Downsample ratio')
    parser.add_argument('--process_length', type=int, default=-1, help='Number of frames to process (-1 = all)')
    parser.add_argument('--process_stride', type=int, default=1, help='Frame stride')

    args = parser.parse_args()

    print("="*60)
    print("Step 1: Geometry Priors Generation (Per-Frame)")
    print("="*60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # PASS 1: Generate raw priors and compute global statistics
    print("\nPass 1: Computing global normalization statistics...")
    pred_point_maps_raw = []
    pred_masks_raw = []

    for i in range(process_length):
        frame = frames_tensor[i:i+1]  # [1, C, H, W]
        pred_p, pred_m = process_frame_priors(prior_model, frame, device='cuda')
        pred_point_maps_raw.append(pred_p)
        pred_masks_raw.append(pred_m)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{process_length} frames")

    # Stack for global statistics
    pred_point_maps_raw = torch.cat(pred_point_maps_raw, dim=0)
    pred_masks_raw = torch.cat(pred_masks_raw, dim=0)

    print("Computing global disparity normalization...")
    min_disparity, max_disparity = postprocess_priors(pred_point_maps_raw, pred_masks_raw)
    print(f"  Min disparity: {min_disparity:.4f}, Max disparity: {max_disparity:.4f}")

    # PASS 2: Normalize and save per-frame
    print("\nPass 2: Normalizing and saving per-frame priors...")
    need_resize = (original_height != height or original_width != width)

    for i in range(process_length):
        # Get raw priors for this frame
        pred_p = pred_point_maps_raw[i:i+1]  # [1, H, W, 3]
        pred_m = pred_masks_raw[i:i+1]      # [1, H, W]

        # Normalize using global statistics
        pred_disp, pred_mask, pred_pmap, pred_intr = normalize_frame(
            pred_p, pred_m, min_disparity, max_disparity
        )

        # Resize if needed
        if need_resize:
            pred_disp = F.interpolate(
                pred_disp.unsqueeze(0).unsqueeze(0), (height, width),
                mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

            pred_mask = F.interpolate(
                pred_mask.unsqueeze(0).unsqueeze(0), (height, width),
                mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)

            # Transform log-depth for interpolation
            pred_pmap = torch.cat([
                F.interpolate(pred_pmap[0:2].unsqueeze(0), (height, width), mode='bilinear', align_corners=False).squeeze(0),
                F.interpolate(
                    pred_pmap[2:3].unsqueeze(0).clamp_max(10).exp(), (height, width),
                    mode='bilinear', align_corners=False
                ).log().squeeze(0)
            ], dim=0)

            pred_intr = F.interpolate(
                pred_intr.unsqueeze(0), (height, width),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        # Save this frame
        frame_path = output_dir / f"frame_{i:05d}_prior.pt"
        torch.save({
            'disparity': pred_disp,      # [H, W]
            'valid_mask': pred_mask,     # [H, W]
            'point_map': pred_pmap,      # [3, H, W]
            'intrinsic_map': pred_intr,  # [4, H, W]
        }, frame_path)

        if (i + 1) % 10 == 0:
            print(f"  Saved {i+1}/{process_length} frames")

    # Free memory
    del pred_point_maps_raw, pred_masks_raw, prior_model
    torch.cuda.empty_cache()

    # Save video info (needed by subsequent steps)
    print(f"\nSaving video info to {args.video_info_path}...")
    torch.save({
        'num_frames': process_length,
        'height': height,
        'width': width,
        'original_height': original_height,
        'original_width': original_width,
        'need_resize': need_resize,
        'downsample_ratio': args.downsample_ratio,
        'per_frame_mode': True,  # Flag to indicate per-frame storage
    }, args.video_info_path)

    print("\nStep 1 complete!")
    print(f"  Saved {process_length} per-frame prior files to {output_dir}")
    print(f"  Frame shape: disparity=[{height}, {width}], point_map=[3, {height}, {width}]")


if __name__ == '__main__':
    main()
