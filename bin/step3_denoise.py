#!/usr/bin/env python3
"""
Step 3: Denoising with UNet (Per-Frame Context Loading)
This is the VRAM-intensive step. Only loads UNet and scheduler.
Loads context files on-demand per sliding window to minimize VRAM usage.
Saves denoised latents to disk and exits to free VRAM.
"""

import argparse
import sys
from pathlib import Path
import torch
from diffusers import EulerDiscreteScheduler
from diffusers.training_utils import set_seed
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geometrycrafter import UNetSpatioTemporalConditionModelVid2vid

offline_root = "/mnt/scratch/mlast/GeometryCrafter/pretrained_models"


def prepare_latents(batch_size, num_frames, num_channels, height, width, dtype, device, generator, latents=None):
    """Prepare initial noise latents."""
    shape = (batch_size, num_frames, num_channels, height // 8, width // 8)

    if latents is None:
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    return latents


def get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size, num_videos_per_prompt, do_cfg):
    """Get additional time IDs for conditioning."""
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

    if do_cfg:
        add_time_ids = torch.cat([add_time_ids, add_time_ids])

    return add_time_ids


def denoise_latents(
    unet,
    scheduler,
    latents_init,
    context_dir,
    added_time_ids,
    num_inference_steps,
    guidance_scale,
    window_size,
    overlap,
    num_frames,
    device,
    dtype
):
    """Perform denoising with sliding window, loading context per-window."""
    stride = window_size - overlap
    latents_all = None
    idx_start = 0

    # Setup weights for overlap blending
    if overlap > 0:
        weights = torch.linspace(0, 1, overlap, device=device)
        weights = weights.view(1, overlap, 1, 1, 1)
    else:
        weights = None

    do_cfg = guidance_scale > 1.0

    print(f"Denoising with window_size={window_size}, overlap={overlap}, stride={stride}")

    while idx_start < num_frames - overlap:
        idx_end = min(idx_start + window_size, num_frames)
        print(f"\nProcessing window: frames {idx_start+1}-{idx_end}/{num_frames}")

        # Reset scheduler for this window
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # Get latents for this window
        latents = latents_init[:, :idx_end - idx_start].clone()

        # Roll latents_init for next window
        latents_init = torch.cat(
            [latents_init[:, -overlap:], latents_init[:, :stride]], dim=1
        )

        # Load context for this window only (on-demand)
        print(f"  Loading context for frames {idx_start}-{idx_end-1}...")
        frame_indices = list(range(idx_start, idx_end))
        video_embeddings_current, video_latents_current, prior_latents_current = load_context_frames(
            context_dir, frame_indices, device, dtype
        )

        # Denoising loop
        with tqdm(total=num_inference_steps, desc=f"Window {idx_start//stride + 1}") as pbar:
            for i, t in enumerate(timesteps):
                # Blend overlapping region on first timestep
                if latents_all is not None and i == 0:
                    latents[:, :overlap] = (
                        latents_all[:, -overlap:]
                        + latents[:, :overlap] / scheduler.init_noise_sigma * scheduler.sigmas[i]
                    )

                # Prepare model input
                latent_model_input = scheduler.scale_model_input(latents, t)
                latent_model_input = torch.cat(
                    [latent_model_input, video_latents_current, prior_latents_current], dim=2
                )

                # Predict noise
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=video_embeddings_current,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # Classifier-free guidance
                if do_cfg:
                    latent_model_input_uncond = scheduler.scale_model_input(latents, t)
                    latent_model_input_uncond = torch.cat(
                        [latent_model_input_uncond,
                         torch.zeros_like(video_latents_current),
                         torch.zeros_like(prior_latents_current)],
                        dim=2,
                    )
                    noise_pred_uncond = unet(
                        latent_model_input_uncond,
                        t,
                        encoder_hidden_states=torch.zeros_like(video_embeddings_current),
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                # Step
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                pbar.update(1)

        # Accumulate results
        if latents_all is None:
            latents_all = latents.clone()
        else:
            if overlap > 0:
                # Blend overlapping region
                latents_all[:, -overlap:] = (
                    latents[:, :overlap] * weights + latents_all[:, -overlap:] * (1 - weights)
                )
            latents_all = torch.cat([latents_all, latents[:, overlap:]], dim=1)

        idx_start += stride

    return latents_all


def load_context_frames(context_dir, frame_indices, device, dtype):
    """Load per-frame context files for specified frame indices."""
    embeddings = []
    vae_latents = []
    prior_latents = []

    for idx in frame_indices:
        # Load embeddings
        embed = torch.load(context_dir / f"frame_{idx:05d}_embed.pt")
        embeddings.append(embed.to(device, dtype=dtype))

        # Load VAE latents
        vae_lat = torch.load(context_dir / f"frame_{idx:05d}_vae_latent.pt")
        vae_latents.append(vae_lat.to(device, dtype=dtype))

        # Load prior latents
        prior_lat = torch.load(context_dir / f"frame_{idx:05d}_prior_latent.pt")
        prior_latents.append(prior_lat.to(device, dtype=dtype))

    # Stack and add batch dimension
    video_embeddings = torch.stack(embeddings, dim=0).unsqueeze(0)  # [1, T, 1024]
    video_latents = torch.stack(vae_latents, dim=0).unsqueeze(0)     # [1, T, C, H, W]
    prior_latents = torch.stack(prior_latents, dim=0).unsqueeze(0)   # [1, T, C, H, W]

    return video_embeddings, video_latents, prior_latents


def main():
    parser = argparse.ArgumentParser(description='Step 3: Denoise Latents with UNet (Per-Frame)')
    parser.add_argument('--context_dir', type=str, required=True, help='Input context directory from Step 2')
    parser.add_argument('--video_info_path', type=str, required=True, help='Input video info .pt file')
    parser.add_argument('--output_path', type=str, required=True, help='Output .pt file for denoised latents')
    parser.add_argument('--cache_dir', type=str, default='workspace/cache', help='Model cache directory')
    parser.add_argument('--num_inference_steps', type=int, default=5, help='Number of denoising steps')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale')
    parser.add_argument('--window_size', type=int, default=110, help='Sliding window size')
    parser.add_argument('--overlap', type=int, default=25, help='Window overlap')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_type', type=str, default='diff', choices=['diff', 'determ'], help='Model type')

    args = parser.parse_args()

    print("="*60)
    print("Step 3: Denoising with UNet (VRAM Intensive) - Per-Frame Mode")
    print("="*60)

    set_seed(args.seed)
    device = 'cuda'
    dtype = torch.float16

    context_dir = Path(args.context_dir)

    # Load video info
    print(f"Loading video info from {args.video_info_path}...")
    video_info = torch.load(args.video_info_path)
    num_frames = video_info['num_frames']
    height = video_info['height']
    width = video_info['width']

    # Adjust window size if needed
    window_size = args.window_size
    overlap = args.overlap
    if num_frames <= window_size:
        window_size = num_frames
        overlap = 0
        print(f"Adjusted window_size={window_size}, overlap={overlap} (video fits in one window)")

    print(f"Will load context per-frame from {context_dir} ({num_frames} frames total)")

    # Load UNet (the big model)
    print(f"\nLoading UNet ({args.model_type})...")
    unet = UNetSpatioTemporalConditionModelVid2vid.from_pretrained(
        f'{offline_root}/TencentARC/GeometryCrafter',
        subfolder=f'unet_{args.model_type}',
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        cache_dir=args.cache_dir
    ).to(device)
    unet.requires_grad_(False)
    print("  UNet loaded.")

    # Load scheduler
    print("Loading scheduler...")
    scheduler = EulerDiscreteScheduler.from_pretrained(
        f"{offline_root}/stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="scheduler",
        cache_dir=args.cache_dir
    )

    # Prepare additional conditioning
    added_time_ids = get_add_time_ids(
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        dtype=dtype,
        batch_size=1,
        num_videos_per_prompt=1,
        do_cfg=args.guidance_scale > 1.0
    ).to(device)

    # Prepare initial latents
    print("\nPreparing initial noise latents...")
    num_channels_latents = 8
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents_init = prepare_latents(
        batch_size=1,
        num_frames=window_size,
        num_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=dtype,
        device=device,
        generator=generator,
        latents=None
    )

    print(f"  Initial latents shape: {list(latents_init.shape)}")

    # Perform denoising
    print("\nStarting denoising...")
    with torch.inference_mode():
        latents_all = denoise_latents(
            unet=unet,
            scheduler=scheduler,
            latents_init=latents_init,
            context_dir=context_dir,
            added_time_ids=added_time_ids,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            window_size=window_size,
            overlap=overlap,
            num_frames=num_frames,
            device=device,
            dtype=dtype
        )

    # Get scaling factor from a temporary VAE config
    # (We need this to properly scale latents, but we don't load the full VAE)
    from diffusers import AutoencoderKLTemporalDecoder
    vae_config = AutoencoderKLTemporalDecoder.load_config(
        f"{offline_root}/stabilityai/stable-video-diffusion-img2vid-xt",
        subfolder="vae",
        cache_dir=args.cache_dir
    )
    scaling_factor = vae_config['scaling_factor']

    # Scale and convert to fp32 for next step
    latents_all = (1 / scaling_factor) * latents_all.squeeze(0).to(torch.float32)

    print(f"\nDenoised latents shape: {list(latents_all.shape)}")

    # Save to disk
    print(f"Saving denoised latents to {args.output_path}...")
    torch.save({
        'latents': latents_all.cpu(),
    }, args.output_path)

    print("Step 3 complete!")


if __name__ == '__main__':
    main()
