#!/usr/bin/env python3
"""
Step 4: Decode to Geometry Maps
Loads PointMapVAE and decodes latents to final geometry representations.
Saves results as .npz file and exits.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from kornia.utils import create_meshgrid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometrycrafter import PMapAutoencoderKLTemporalDecoder

offline_root = "/mnt/scratch/mlast/GeometryCrafter/pretrained_models"


def decode_point_map(
    point_map_vae,
    latents,
    chunk_size=8,
    force_projection=True,
    force_fixed_focal=True,
    use_extract_interp=False,
    need_resize=False,
    height=None,
    width=None,
    device='cuda'
):
    """Decode latents to point maps and valid masks."""
    T = latents.shape[0]
    rec_intrinsic_maps = []
    rec_depth_maps = []
    rec_valid_masks = []

    print(f"Decoding {T} frames in chunks of {chunk_size}...")
    for i in range(0, T, chunk_size):
        end_idx = min(i + chunk_size, T)
        lat = latents[i:end_idx].to(device)

        rec_imap, rec_dmap, rec_vmask = point_map_vae.decode(
            lat,
            num_frames=lat.shape[0],
        )

        rec_intrinsic_maps.append(rec_imap.cpu())
        rec_depth_maps.append(rec_dmap.cpu())
        rec_valid_masks.append(rec_vmask.cpu())

        print(f"  Decoded frames {i+1}-{end_idx}/{T}")

    rec_intrinsic_maps = torch.cat(rec_intrinsic_maps, dim=0)
    rec_depth_maps = torch.cat(rec_depth_maps, dim=0)
    rec_valid_masks = torch.cat(rec_valid_masks, dim=0)

    # Resize if needed
    if need_resize:
        print(f"Resizing outputs to {height}x{width}...")
        if use_extract_interp:
            rec_depth_maps = F.interpolate(rec_depth_maps, (height, width), mode='nearest-exact')
            rec_valid_masks = F.interpolate(rec_valid_masks, (height, width), mode='nearest-exact')
        else:
            # Transform log-depth to depth domain for interpolation
            rec_depth_maps = F.interpolate(
                rec_depth_maps.clamp_max(10).exp(), (height, width),
                mode='bilinear', align_corners=False
            ).log()
            rec_valid_masks = F.interpolate(
                rec_valid_masks, (height, width),
                mode='bilinear', align_corners=False
            )
        rec_intrinsic_maps = F.interpolate(
            rec_intrinsic_maps, (height, width),
            mode='bilinear', align_corners=False
        )

    H, W = rec_intrinsic_maps.shape[-2], rec_intrinsic_maps.shape[-1]

    # Create mesh grid
    mesh_grid = create_meshgrid(H, W, normalized_coordinates=True).to(
        rec_intrinsic_maps.device, rec_intrinsic_maps.dtype, non_blocking=True
    )  # 1,H,W,2

    # Scale intrinsic maps
    rec_intrinsic_maps = torch.cat([
        rec_intrinsic_maps * W / np.sqrt(W**2 + H**2),
        rec_intrinsic_maps * H / np.sqrt(W**2 + H**2)
    ], dim=1)  # T,2,H,W

    mesh_grid = mesh_grid.permute(0, 3, 1, 2)  # 1,2,H,W
    rec_valid_masks = rec_valid_masks.squeeze(1) > 0  # T,H,W

    # Force projection (ensure consistent focal length)
    if force_projection:
        if force_fixed_focal:
            # Use global average focal length
            nfx = (rec_intrinsic_maps[:, 0, :, :] * rec_valid_masks.float()).mean() / (rec_valid_masks.float().mean() + 1e-4)
            nfy = (rec_intrinsic_maps[:, 1, :, :] * rec_valid_masks.float()).mean() / (rec_valid_masks.float().mean() + 1e-4)
            rec_intrinsic_maps = torch.tensor([nfx, nfy], device=rec_intrinsic_maps.device)[None, :, None, None].repeat(T, 1, 1, 1)
        else:
            # Use per-frame average focal length
            nfx = (rec_intrinsic_maps[:, 0, :, :] * rec_valid_masks.float()).mean(dim=[-1, -2]) / (rec_valid_masks.float().mean(dim=[-1, -2]) + 1e-4)
            nfy = (rec_intrinsic_maps[:, 1, :, :] * rec_valid_masks.float()).mean(dim=[-1, -2]) / (rec_valid_masks.float().mean(dim=[-1, -2]) + 1e-4)
            rec_intrinsic_maps = torch.stack([nfx, nfy], dim=-1)[:, :, None, None]  # T,2,1,1

    # Reconstruct point maps
    rec_point_maps = torch.cat([rec_intrinsic_maps * mesh_grid, rec_depth_maps], dim=1).permute(0, 2, 3, 1)  # T,H,W,3
    xy, z = rec_point_maps.split([2, 1], dim=-1)
    z = torch.clamp_max(z, 10)  # Numerical stability
    z = torch.exp(z)
    rec_point_maps = torch.cat([xy * z, z], dim=-1)

    return rec_point_maps, rec_valid_masks


def main():
    parser = argparse.ArgumentParser(description='Step 4: Decode to Geometry Maps')
    parser.add_argument('--denoised_path', type=str, required=True, help='Input denoised latents .pt file from Step 3')
    parser.add_argument('--video_info_path', type=str, required=True, help='Input video info .pt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--video_basename', type=str, required=True, help='Video basename for output file')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')
    parser.add_argument('--decode_chunk_size', type=int, default=8, help='Chunk size for decoding')
    parser.add_argument('--force_projection', type=str, default='true', help='Force projection')
    parser.add_argument('--force_fixed_focal', type=str, default='true', help='Force fixed focal length')
    parser.add_argument('--use_extract_interp', type=str, default='false', help='Use exact interpolation')

    args = parser.parse_args()

    # Parse boolean arguments
    force_projection = args.force_projection.lower() == 'true'
    force_fixed_focal = args.force_fixed_focal.lower() == 'true'
    use_extract_interp = args.use_extract_interp.lower() == 'true'

    print("="*60)
    print("Step 4: Decoding to Geometry Maps")
    print("="*60)

    device = 'cuda'

    # Load video info
    print(f"Loading video info from {args.video_info_path}...")
    video_info = torch.load(args.video_info_path)
    original_height = video_info['original_height']
    original_width = video_info['original_width']
    need_resize = video_info['need_resize']
    downsample_ratio = video_info.get('downsample_ratio', 1.0)

    # Load denoised latents
    print(f"Loading denoised latents from {args.denoised_path}...")
    denoised = torch.load(args.denoised_path)
    latents = denoised['latents']
    print(f"  Latents shape: {list(latents.shape)}")

    # Load PointMapVAE
    print(f"Loading PointMapVAE...")
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        f'{offline_root}/TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir
    ).to(device)
    point_map_vae.requires_grad_(False)
    print("  PointMapVAE loaded.")

    # Decode
    print("\nDecoding latents to point maps...")
    with torch.inference_mode():
        rec_point_map, rec_valid_mask = decode_point_map(
            point_map_vae=point_map_vae,
            latents=latents,
            chunk_size=args.decode_chunk_size,
            force_projection=force_projection,
            force_fixed_focal=force_fixed_focal,
            use_extract_interp=use_extract_interp,
            need_resize=need_resize,
            height=original_height,
            width=original_width,
            device=device
        )

    # If downsampled, upsample back to original resolution
    if downsample_ratio > 1.0:
        print(f"Upsampling back to original resolution {original_height}x{original_width}...")
        rec_point_map = F.interpolate(
            rec_point_map.permute(0, 3, 1, 2), (original_height, original_width),
            mode='bilinear'
        ).permute(0, 2, 3, 1)
        rec_valid_mask = F.interpolate(
            rec_valid_mask.float().unsqueeze(1), (original_height, original_width),
            mode='bilinear'
        ).squeeze(1) > 0.5

    print(f"Final point map shape: {list(rec_point_map.shape)}")
    print(f"Final valid mask shape: {list(rec_valid_mask.shape)}")

    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.video_basename}.npz"

    print(f"\nSaving output to {output_path}...")
    np.savez(
        str(output_path),
        point_map=rec_point_map.detach().cpu().numpy().astype(np.float16),
        mask=rec_valid_mask.detach().cpu().numpy().astype(np.bool_)
    )

    print("Step 4 complete!")
    print(f"Output saved: {output_path}")


if __name__ == '__main__':
    main()
