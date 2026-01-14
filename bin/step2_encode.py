#!/usr/bin/env python3
"""
Step 2: Context Encoding (Per-Frame)
Processes each frame individually through CLIP, VAE, and PointMapVAE.
Saves per-frame latents to minimize VRAM usage.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from decord import VideoReader, cpu
from diffusers import AutoencoderKLTemporalDecoder
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometrycrafter import PMapAutoencoderKLTemporalDecoder


def resize_with_antialiasing(image, size):
    """Resize with antialiasing (from diffusers)."""
    return F.interpolate(image, size, mode='bicubic', align_corners=False, antialias=True)


def encode_frame_embedding(frame, image_encoder, feature_extractor):
    """Encode a single frame to CLIP embedding."""
    # Resize to 224x224 for CLIP
    frame_224 = resize_with_antialiasing(frame.float(), (224, 224))
    frame_224 = (frame_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    # Preprocess and encode
    inputs = feature_extractor(
        images=frame_224,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values.to(frame.device, dtype=frame.dtype)

    emb = image_encoder(inputs).image_embeds
    return emb.cpu()


def encode_frame_vae(frame, vae):
    """Encode a single frame to VAE latent."""
    latent = vae.encode(frame.to(vae.dtype)).latent_dist.mode()
    return latent.cpu()


def encode_frame_prior(disparity, valid_mask, point_map_z, intrinsic_map_focal, point_map_vae, vae, device):
    """Encode geometry prior for a single frame using PointMapVAE."""
    # disparity: [H, W]
    # valid_mask: [H, W]
    # point_map_z: [1, H, W] - only Z channel
    # intrinsic_map_focal: [2, H, W] - only focal components

    # Create pseudo-image from disparity
    pseudo_image = disparity.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1, 3, H, W]

    # Extract focal length magnitude from intrinsic map (already subset to [2:4])
    intrinsic_scalar = torch.norm(intrinsic_map_focal, p=2, dim=0, keepdim=False)  # [H, W]

    # First encode pseudo-image with VAE
    latent_dist = vae.encode(pseudo_image.to(device, dtype=vae.dtype)).latent_dist

    # Then encode with PointMapVAE
    prior_input = torch.cat([
        intrinsic_scalar.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        point_map_z.unsqueeze(0),                     # [1, 1, H, W]
        disparity.unsqueeze(0).unsqueeze(0),          # [1, 1, H, W]
        valid_mask.unsqueeze(0).unsqueeze(0),         # [1, 1, H, W]
    ], dim=1).to(device)  # [1, 4, H, W]

    latent_dist = point_map_vae.encode(prior_input, latent_dist)

    if hasattr(latent_dist, 'mode'):
        latent = latent_dist.mode()
    else:
        latent = latent_dist

    latent = latent * vae.config.scaling_factor
    return latent.squeeze(0).cpu()  # [C, H/8, W/8]


def main():
    parser = argparse.ArgumentParser(description='Step 2: Encode Context (Per-Frame)')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--priors_dir', type=str, required=True, help='Input directory with per-frame priors from Step 1')
    parser.add_argument('--video_info_path', type=str, required=True, help='Input video info .pt file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for per-frame context')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')

    args = parser.parse_args()

    print("="*60)
    print("Step 2: Context Encoding (Per-Frame)")
    print("="*60)

    device = 'cuda'
    dtype = torch.float16

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    priors_dir = Path(args.priors_dir)

    # Load video info
    print(f"Loading video info from {args.video_info_path}...")
    video_info = torch.load(args.video_info_path)
    num_frames = video_info['num_frames']
    height = video_info['height']
    width = video_info['width']
    need_resize = video_info['need_resize']
    downsample_ratio = video_info['downsample_ratio']

    print(f"  Processing {num_frames} frames at {height}x{width}")

    # Load video frames
    print(f"Loading video: {args.video_path}")
    vid = VideoReader(args.video_path, ctx=cpu(0))
    frames_idx = list(range(0, len(vid), video_info.get('process_stride', 1)))
    frames = vid.get_batch(frames_idx[:num_frames]).asnumpy().astype(np.float32) / 255.0
    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)

    # Apply downsampling if needed
    if downsample_ratio > 1.0:
        new_h = round(frames_tensor.shape[-2] / downsample_ratio)
        new_w = round(frames_tensor.shape[-1] / downsample_ratio)
        frames_tensor = F.interpolate(
            frames_tensor, (new_h, new_w),
            mode='bicubic', antialias=True
        ).clamp(0, 1)

    # Resize to target dimensions
    if need_resize:
        frames_tensor = F.interpolate(
            frames_tensor, (height, width),
            mode='bicubic', align_corners=False, antialias=True
        ).clamp(0, 1)

    # Convert to [-1, 1] range
    frames_tensor = frames_tensor * 2.0 - 1.0

    print(f"Video tensor shape: {list(frames_tensor.shape)}")

    # ==================================================
    # ENCODING PART 1: Image Embeddings (CLIP) - Per Frame
    # ==================================================
    print("\n[Encoding 1/3] Loading Image Encoder (CLIP)...")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="feature_extractor",
        cache_dir=args.cache_dir
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="image_encoder",
        torch_dtype=dtype,
        cache_dir=args.cache_dir
    ).to(device)

    print("  Generating per-frame video embeddings...")
    for i in range(num_frames):
        frame = frames_tensor[i:i+1].to(device, dtype=dtype)  # [1, C, H, W]

        with torch.inference_mode():
            embedding = encode_frame_embedding(frame, image_encoder, feature_extractor)

        # Save immediately
        embed_path = output_dir / f"frame_{i:05d}_embed.pt"
        torch.save(embedding, embed_path)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{num_frames} frames")

    # Free memory
    del image_encoder, feature_extractor
    torch.cuda.empty_cache()
    print("  CLIP encoder unloaded")

    # ==================================================
    # ENCODING PART 2: VAE Latents - Per Frame
    # ==================================================
    print("\n[Encoding 2/3] Loading VAE...")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="vae",
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir
    ).to(device)

    # Handle upcast if needed
    needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
    if needs_upcasting:
        vae.to(dtype=torch.float32)

    print("  Encoding per-frame VAE latents...")
    for i in range(num_frames):
        frame = frames_tensor[i:i+1].to(device, dtype=vae.dtype)  # [1, C, H, W]

        with torch.inference_mode():
            latent = encode_frame_vae(frame, vae)

        # Save immediately
        latent_path = output_dir / f"frame_{i:05d}_vae_latent.pt"
        torch.save(latent, latent_path)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{num_frames} frames")

    # Don't free VAE yet - needed for PointMapVAE encoding

    # ==================================================
    # ENCODING PART 3: Prior Latents (PointMapVAE) - Per Frame
    # ==================================================
    print("\n[Encoding 3/3] Loading PointMapVAE...")
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir
    ).to(device)

    print("  Encoding per-frame prior latents...")
    for i in range(num_frames):
        # Load priors for this frame
        prior_path = priors_dir / f"frame_{i:05d}_prior.pt"
        prior_data = torch.load(prior_path)

        pred_disparity = prior_data['disparity']              # [H, W]
        pred_valid_mask = prior_data['valid_mask']            # [H, W]
        pred_point_map_z = prior_data['point_map_z']          # [1, H, W]
        pred_intrinsic_map_focal = prior_data['intrinsic_map_focal']  # [2, H, W]

        with torch.inference_mode():
            prior_latent = encode_frame_prior(
                pred_disparity, pred_valid_mask, pred_point_map_z, pred_intrinsic_map_focal,
                point_map_vae, vae, device
            )

        # Save immediately
        prior_latent_path = output_dir / f"frame_{i:05d}_prior_latent.pt"
        torch.save(prior_latent, prior_latent_path)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{num_frames} frames")

    # Cast VAE back if needed
    if needs_upcasting:
        vae.to(dtype=torch.float16)

    # Free memory
    del vae, point_map_vae
    torch.cuda.empty_cache()
    print("  VAE and PointMapVAE unloaded")

    print("\nStep 2 complete!")
    print(f"  Saved {num_frames} per-frame encoding files to {output_dir}")
    print(f"  Files per frame:")
    print(f"    - frame_XXXXX_embed.pt (CLIP embedding)")
    print(f"    - frame_XXXXX_vae_latent.pt (VAE latent)")
    print(f"    - frame_XXXXX_prior_latent.pt (Prior latent)")


if __name__ == '__main__':
    main()
