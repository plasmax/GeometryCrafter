import numpy as np
import cv2
import argparse
from pathlib import Path


def npz_to_mp4(npz_path, output_path, fps=30):
    """Convert .npz file with point_map and mask to MP4 video."""

    # Load the npz file
    data = np.load(npz_path)
    point_map = data['point_map']  # Shape: (T, H, W, 3)
    mask = data['mask']  # Shape: (T, H, W)

    T, H, W, _ = point_map.shape

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    print(f"Converting {npz_path} to {output_path}")
    print(f"Video shape: {T} frames, {H}x{W}")

    for i in range(T):
        # Get depth (z-coordinate) from point map
        depth = point_map[i, :, :, 2]
        valid_mask = mask[i]

        # Normalize depth for visualization (only valid pixels)
        valid_depth = depth[valid_mask]
        if len(valid_depth) > 0:
            depth_min, depth_max = valid_depth.min(), valid_depth.max()
            if depth_max > depth_min:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth)
        else:
            depth_norm = np.zeros_like(depth)

        # Apply mask (set invalid pixels to black)
        depth_norm[~valid_mask] = 0

        # Convert to 8-bit and apply colormap
        depth_vis = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

        # Write frame
        out.write(depth_color)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{T} frames")

    out.release()
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .npz to MP4 video")
    parser.add_argument("--npz_path", type=str, default="workspace/examples_output/video4.npz",
                        help="Path to input .npz file")
    parser.add_argument("--output_path", type=str, default="workspace/examples_output/video4_depth.mp4",
                        help="Path to output MP4 file")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS")

    args = parser.parse_args()

    npz_to_mp4(args.npz_path, args.output_path, args.fps)
