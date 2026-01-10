#!/usr/bin/env python3
"""
Step 2: Context Encoding
Loads Image Encoder, VAE, and PointMapVAE sequentially (not simultaneously).
Generates video embeddings, video latents, and prior latents.
Saves results to disk and exits to free VRAM.
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


def encode_video_embeddings(video, image_encoder, feature_extractor, chunk_size=14):
    """Encode video frames to image embeddings using CLIP."""
    # Resize to 224x224 for CLIP
    video_224 = resize_with_antialiasing(video.float(), (224, 224))
    video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

    embeddings = []
    for i in range(0, video_224.shape[0], chunk_size):
        end_idx = min(i + chunk_size, video_224.shape[0])
        chunk = video_224[i:end_idx]

        # Preprocess and encode
        inputs = feature_extractor(
            images=chunk,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values.to(video.device, dtype=video.dtype)

        emb = image_encoder(inputs).image_embeds
        embeddings.append(emb)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def encode_vae_video(video, vae, chunk_size=14):
    """Encode video frames to VAE latents."""
    video_latents = []
    for i in range(0, video.shape[0], chunk_size):
        end_idx = min(i + chunk_size, video.shape[0])
        chunk = video[i:end_idx]

        latent = vae.encode(chunk).latent_dist.mode()
        video_latents.append(latent)

    video_latents = torch.cat(video_latents, dim=0)
    return video_latents


def encode_point_map(point_map_vae, vae, disparity, valid_mask, point_map, intrinsic_map, chunk_size=8):
    """Encode geometry priors to latent space using PointMapVAE."""
    T, _, H, W = point_map.shape
    latents = []

    # Create pseudo-image from disparity
    pseudo_image = disparity[:, None].repeat(1, 3, 1, 1)

    # Extract focal length magnitude from intrinsic map
    intrinsic_map = torch.norm(intrinsic_map[:, 2:4], p=2, dim=1, keepdim=False)

    for i in range(0, T, chunk_size):
        end_idx = min(i + chunk_size, T)

        # First encode pseudo-image with VAE
        latent_dist = vae.encode(pseudo_image[i:end_idx].to(dtype=vae.dtype)).latent_dist

        # Then encode with PointMapVAE
        latent_dist = point_map_vae.encode(
            torch.cat([
                intrinsic_map[i:end_idx, None],
                point_map[i:end_idx, 2:3],
                disparity[i:end_idx, None],
                valid_mask[i:end_idx, None],
            ], dim=1),
            latent_dist
        )

        if hasattr(latent_dist, 'mode'):
            latent = latent_dist.mode()
        else:
            latent = latent_dist

        latents.append(latent)

    latents = torch.cat(latents, dim=0)
    latents = latents * vae.config.scaling_factor
    return latents


def main():
    parser = argparse.ArgumentParser(description='Step 2: Encode Context')
    parser.add_argument('--video_path', type=str, required=True, help='Input video path')
    parser.add_argument('--priors_path', type=str, required=True, help='Input priors .pt file from Step 1')
    parser.add_argument('--video_info_path', type=str, required=True, help='Input video info .pt file')
    parser.add_argument('--output_path', type=str, required=True, help='Output .pt file for context')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')
    parser.add_argument('--decode_chunk_size', type=int, default=8, help='Chunk size for processing')

    args = parser.parse_args()

    print("="*60)
    print("Step 2: Context Encoding")
    print("="*60)

    device = 'cuda'
    dtype = torch.float16

    # Load video info
    print(f"Loading video info from {args.video_info_path}...")
    video_info = torch.load(args.video_info_path)
    num_frames = video_info['num_frames']
    height = video_info['height']
    width = video_info['width']
    need_resize = video_info['need_resize']
    downsample_ratio = video_info['downsample_ratio']

    # Load priors
    print(f"Loading priors from {args.priors_path}...")
    priors = torch.load(args.priors_path)
    pred_disparity = priors['disparity']
    pred_valid_mask = priors['valid_mask']
    pred_point_map = priors['point_map']
    pred_intrinsic_map = priors['intrinsic_map']
    print(f"  Loaded priors: {list(pred_disparity.shape)}")

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
    video = frames_tensor.to(device=device, dtype=dtype)
    video = video * 2.0 - 1.0

    print(f"Video tensor shape: {list(video.shape)}")

    # ==================================================
    # ENCODING PART 1: Image Embeddings (CLIP)
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

    print("  Generating video embeddings...")
    with torch.inference_mode():
        video_embeddings = encode_video_embeddings(
            video, image_encoder, feature_extractor,
            chunk_size=args.decode_chunk_size
        ).unsqueeze(0)  # [1, T, 1024]

    print(f"  Video embeddings shape: {list(video_embeddings.shape)}")

    # Free memory
    del image_encoder, feature_extractor
    torch.cuda.empty_cache()

    # ==================================================
    # ENCODING PART 2: VAE Latents
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

    print("  Encoding video to VAE latents...")
    with torch.inference_mode():
        video_latents = encode_vae_video(
            video.to(vae.dtype), vae,
            chunk_size=args.decode_chunk_size
        ).unsqueeze(0).to(video_embeddings.dtype)  # [1, T, C, H, W]

    print(f"  Video latents shape: {list(video_latents.shape)}")

    # Don't free VAE yet - needed for PointMapVAE encoding

    # ==================================================
    # ENCODING PART 3: Prior Latents (PointMapVAE)
    # ==================================================
    print("\n[Encoding 3/3] Loading PointMapVAE...")
    point_map_vae = PMapAutoencoderKLTemporalDecoder.from_pretrained(
        'TencentARC/GeometryCrafter',
        subfolder='point_map_vae',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
        cache_dir=args.cache_dir
    ).to(device)

    print("  Encoding geometry priors to latent space...")
    with torch.inference_mode():
        prior_latents = encode_point_map(
            point_map_vae, vae,
            pred_disparity.to(device),
            pred_valid_mask.to(device),
            pred_point_map.to(device),
            pred_intrinsic_map.to(device),
            chunk_size=args.decode_chunk_size
        ).unsqueeze(0).to(video_embeddings.dtype)  # [1, T, C, H, W]

    print(f"  Prior latents shape: {list(prior_latents.shape)}")

    # Cast VAE back if needed
    if needs_upcasting:
        vae.to(dtype=torch.float16)

    # Free memory
    del vae, point_map_vae
    torch.cuda.empty_cache()

    # ==================================================
    # Save Context to Disk
    # ==================================================
    print(f"\nSaving context to {args.output_path}...")
    torch.save({
        'video_embeddings': video_embeddings.cpu(),
        'video_latents': video_latents.cpu(),
        'prior_latents': prior_latents.cpu(),
    }, args.output_path)

    print("Step 2 complete!")
    print(f"  Embeddings: {list(video_embeddings.shape)}")
    print(f"  Video latents: {list(video_latents.shape)}")
    print(f"  Prior latents: {list(prior_latents.shape)}")


if __name__ == '__main__':
    main()
