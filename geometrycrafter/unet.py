from typing import Union, Tuple, Optional, List
import os
import tempfile

import torch
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils import is_torch_version


class ResidualCache:
    """Cache for UNet residual connections to reduce VRAM usage.

    Supports two modes:
    - 'cpu': Store residuals in CPU memory (faster, uses system RAM)
    - 'disk': Store residuals on disk (slower, minimal memory usage)

    Residuals are stored during the down-sampling pass and retrieved during
    up-sampling, allowing VRAM to be freed between passes.
    """

    def __init__(self, mode: str = "cpu", cache_dir: Optional[str] = None, pin_memory: bool = True):
        """Initialize residual cache.

        Args:
            mode: 'cpu' for CPU memory, 'disk' for disk storage
            cache_dir: Directory for disk cache (auto-created if None)
            pin_memory: Use pinned memory for faster GPU transfers (cpu mode only)
        """
        assert mode in ("cpu", "disk"), f"Invalid mode: {mode}"
        self.mode = mode
        self.pin_memory = pin_memory and mode == "cpu" and torch.cuda.is_available()
        self._storage: List = []  # List of residual tuples (cpu mode) or file paths (disk mode)
        self._cache_dir = cache_dir
        self._temp_dir = None

        if mode == "disk":
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                self._cache_dir = cache_dir
            else:
                self._temp_dir = tempfile.mkdtemp(prefix="unet_residuals_")
                self._cache_dir = self._temp_dir

    def save(self, res_samples: Tuple[torch.Tensor, ...]) -> None:
        """Save residual samples, freeing GPU memory."""
        if self.mode == "cpu":
            # Move to CPU, optionally pinned for faster transfers back
            if self.pin_memory:
                cpu_samples = tuple(
                    t.detach().cpu().pin_memory() for t in res_samples
                )
            else:
                cpu_samples = tuple(t.detach().cpu() for t in res_samples)
            self._storage.append(cpu_samples)
        else:
            # Save to disk
            idx = len(self._storage)
            path = os.path.join(self._cache_dir, f"res_{idx}.pt")
            torch.save(tuple(t.detach().cpu() for t in res_samples), path)
            self._storage.append(path)

    def load(self, idx: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Load residual samples back to GPU."""
        if self.mode == "cpu":
            cpu_samples = self._storage[idx]
            return tuple(t.to(device=device, dtype=dtype, non_blocking=self.pin_memory) for t in cpu_samples)
        else:
            path = self._storage[idx]
            cpu_samples = torch.load(path, weights_only=True)
            return tuple(t.to(device=device, dtype=dtype) for t in cpu_samples)

    def load_reverse(self, reverse_idx: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, ...]:
        """Load residuals in reverse order (for up-sampling pass)."""
        actual_idx = len(self._storage) - 1 - reverse_idx
        return self.load(actual_idx, device, dtype)

    def clear(self) -> None:
        """Clear all cached residuals and free memory."""
        if self.mode == "disk":
            for path in self._storage:
                if os.path.exists(path):
                    os.remove(path)
        self._storage.clear()

    def cleanup(self) -> None:
        """Clean up resources (call when done with forward pass)."""
        self.clear()
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                os.rmdir(self._temp_dir)
            except OSError:
                pass  # Directory not empty or already removed

    def __len__(self) -> int:
        return len(self._storage)

    def __del__(self):
        self.cleanup()


class UNetSpatioTemporalConditionModelVid2vid(
    UNetSpatioTemporalConditionModel
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _iter_offloadable_modules(self):
        return [
            self.conv_in,
            *self.down_blocks,
            self.mid_block,
            *self.up_blocks,
            self.conv_norm_out,
            self.conv_out,
        ]

    def enable_block_cpu_offload(self, main_device="cuda", offload_device="cpu"):
        """Lightweight weight offload: keep blocks on CPU, stream to GPU one-at-a-time.

        This mirrors sequential CPU offload without requiring accelerate.ModelHook.
        """
        self._block_cpu_offload = True
        self._main_device = torch.device(main_device)
        self._offload_device = torch.device(offload_device)

        # Move heavy blocks to offload device up front to keep VRAM free.
        for module in self._iter_offloadable_modules():
            module.to(self._offload_device)

        # Keep small time embedding layers on main device to avoid repeated transfers.
        for small in (self.time_proj, self.time_embedding, self.add_time_proj, self.add_embedding):
            small.to(self._main_device)

    def enable_residual_offload(
        self,
        mode: str = "cpu",
        cache_dir: Optional[str] = None,
        pin_memory: bool = True
    ):
        """Enable offloading of residual connections to reduce VRAM.

        During the forward pass, residual tensors from down-blocks are moved
        off-GPU and reloaded during up-blocks. This significantly reduces peak
        VRAM usage at the cost of some speed.

        Args:
            mode: 'cpu' for CPU memory (faster), 'disk' for disk storage (minimal RAM)
            cache_dir: Directory for disk cache. Only used if mode='disk'.
            pin_memory: Use pinned memory for faster CPU<->GPU transfers (cpu mode only)

        Example:
            unet.enable_residual_offload(mode='cpu')  # Use CPU RAM
            unet.enable_residual_offload(mode='disk', cache_dir='/tmp/residuals')  # Use disk
        """
        self._residual_offload_mode = mode
        self._residual_offload_cache_dir = cache_dir
        self._residual_offload_pin_memory = pin_memory

    def disable_residual_offload(self):
        """Disable residual offloading (keep residuals in VRAM)."""
        self._residual_offload_mode = None
        self._residual_offload_cache_dir = None
        self._residual_offload_pin_memory = True

    def _create_residual_cache(self) -> Optional[ResidualCache]:
        """Create a ResidualCache if offloading is enabled."""
        mode = getattr(self, "_residual_offload_mode", None)
        if mode is None:
            return None
        return ResidualCache(
            mode=mode,
            cache_dir=getattr(self, "_residual_offload_cache_dir", None),
            pin_memory=getattr(self, "_residual_offload_pin_memory", True),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.conv_in.weight.dtype)

        emb = self.time_embedding(t_emb)  # [batch_size * num_frames, channels]

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, frames, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.flatten(0, 1).unsqueeze(1)

        # 2. pre-process
        block_offload = getattr(self, "_block_cpu_offload", False)
        main_device = getattr(self, "_main_device", sample.device)
        offload_device = getattr(self, "_offload_device", torch.device("cpu"))

        def _to_main(module):
            if block_offload:
                module.to(main_device)

        def _to_offload(module):
            if block_offload:
                module.to(offload_device)
                if main_device.type == "cuda":
                    torch.cuda.empty_cache()

        _to_main(self.conv_in)
        sample = sample.to(dtype=self.conv_in.weight.dtype)
        assert sample.dtype == self.conv_in.weight.dtype, (
            f"sample.dtype: {sample.dtype}, "
            f"self.conv_in.weight.dtype: {self.conv_in.weight.dtype}"
        )
        sample = self.conv_in(sample)
        _to_offload(self.conv_in)

        image_only_indicator = torch.zeros(
            batch_size, num_frames, dtype=sample.dtype, device=sample.device
        )

        # Setup residual caching if enabled
        residual_cache = self._create_residual_cache()
        use_residual_cache = residual_cache is not None

        # For caching: store individual residuals, not per-block tuples
        cached_residuals: List = [] if use_residual_cache else None

        # Initial residual - cache it or keep in tuple
        if use_residual_cache:
            residual_cache.save((sample,))
            cached_residuals.append(None)
            down_block_res_samples = None
        else:
            down_block_res_samples = (sample,)

        if self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):

                for downsample_block in self.down_blocks:
                    _to_main(downsample_block)
                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    else:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    _to_offload(downsample_block)

                    # Save residuals to cache or keep in tuple
                    if use_residual_cache:
                        # Store each residual individually for flat indexing
                        for res in res_samples:
                            residual_cache.save((res,))  # Save as single-element tuple
                            cached_residuals.append(None)  # Placeholder for tracking count
                        del res_samples
                        if main_device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        down_block_res_samples += res_samples

                # 4. mid
                _to_main(self.mid_block)
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    emb,
                    encoder_hidden_states,
                    image_only_indicator,
                    use_reentrant=False,
                )
                _to_offload(self.mid_block)

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    # Load residuals from cache or get from tuple
                    if use_residual_cache:
                        # Load the required number of residuals from the end of the cache
                        num_residuals = len(upsample_block.resnets)
                        res_samples = []
                        for _ in range(num_residuals):
                            # Load from cache in reverse order
                            idx = len(cached_residuals) - 1
                            cached_residuals.pop()
                            res_tuple = residual_cache.load(idx, sample.device, sample.dtype)
                            res_samples.insert(0, res_tuple[0])  # Extract single tensor from tuple
                        res_samples = tuple(res_samples)
                    else:
                        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                        down_block_res_samples = down_block_res_samples[
                            : -len(upsample_block.resnets)
                        ]

                    _to_main(upsample_block)
                    if (
                        hasattr(upsample_block, "has_cross_attention")
                        and upsample_block.has_cross_attention
                    ):
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    else:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            image_only_indicator,
                            use_reentrant=False,
                        )
                    _to_offload(upsample_block)

                    # Free loaded residuals
                    if use_residual_cache:
                        del res_samples
                        if main_device.type == "cuda":
                            torch.cuda.empty_cache()
            else:

                for downsample_block in self.down_blocks:
                    _to_main(downsample_block)
                    if (
                        hasattr(downsample_block, "has_cross_attention")
                        and downsample_block.has_cross_attention
                    ):
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                        )
                    else:
                        sample, res_samples = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(downsample_block),
                            sample,
                            emb,
                            image_only_indicator,
                        )
                    _to_offload(downsample_block)

                    # Save residuals to cache or keep in tuple
                    if use_residual_cache:
                        # Store each residual individually for flat indexing
                        for res in res_samples:
                            residual_cache.save((res,))  # Save as single-element tuple
                            cached_residuals.append(None)  # Placeholder for tracking count
                        del res_samples
                        if main_device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        down_block_res_samples += res_samples

                # 4. mid
                _to_main(self.mid_block)
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    emb,
                    encoder_hidden_states,
                    image_only_indicator,
                )
                _to_offload(self.mid_block)

                # 5. up
                for i, upsample_block in enumerate(self.up_blocks):
                    # Load residuals from cache or get from tuple
                    if use_residual_cache:
                        # Load the required number of residuals from the end of the cache
                        num_residuals = len(upsample_block.resnets)
                        res_samples = []
                        for _ in range(num_residuals):
                            # Load from cache in reverse order
                            idx = len(cached_residuals) - 1
                            cached_residuals.pop()
                            res_tuple = residual_cache.load(idx, sample.device, sample.dtype)
                            res_samples.insert(0, res_tuple[0])  # Extract single tensor from tuple
                        res_samples = tuple(res_samples)
                    else:
                        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                        down_block_res_samples = down_block_res_samples[
                            : -len(upsample_block.resnets)
                        ]

                    _to_main(upsample_block)
                    if (
                        hasattr(upsample_block, "has_cross_attention")
                        and upsample_block.has_cross_attention
                    ):
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            encoder_hidden_states,
                            image_only_indicator,
                        )
                    else:
                        sample = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(upsample_block),
                            sample,
                            res_samples,
                            emb,
                            image_only_indicator,
                        )
                    _to_offload(upsample_block)

                    # Free loaded residuals
                    if use_residual_cache:
                        del res_samples
                        if main_device.type == "cuda":
                            torch.cuda.empty_cache()

        else:
            for downsample_block in self.down_blocks:
                _to_main(downsample_block)
                if (
                    hasattr(downsample_block, "has_cross_attention")
                    and downsample_block.has_cross_attention
                ):
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )

                else:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )

                _to_offload(downsample_block)

                # Save residuals to cache or keep in tuple
                if use_residual_cache:
                    # Store each residual individually for flat indexing
                    for res in res_samples:
                        residual_cache.save((res,))  # Save as single-element tuple
                        cached_residuals.append(None)  # Placeholder for tracking count
                    del res_samples
                    if main_device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    down_block_res_samples += res_samples

            # 4. mid
            _to_main(self.mid_block)
            sample = self.mid_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
            )
            _to_offload(self.mid_block)

            # 5. up
            for i, upsample_block in enumerate(self.up_blocks):
                # Load residuals from cache or get from tuple
                if use_residual_cache:
                    # Load the required number of residuals from the end of the cache
                    num_residuals = len(upsample_block.resnets)
                    res_samples = []
                    for _ in range(num_residuals):
                        # Load from cache in reverse order
                        idx = len(cached_residuals) - 1
                        cached_residuals.pop()
                        res_tuple = residual_cache.load(idx, sample.device, sample.dtype)
                        res_samples.insert(0, res_tuple[0])  # Extract single tensor from tuple
                    res_samples = tuple(res_samples)
                else:
                    res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                    down_block_res_samples = down_block_res_samples[
                        : -len(upsample_block.resnets)
                    ]

                _to_main(upsample_block)
                if (
                    hasattr(upsample_block, "has_cross_attention")
                    and upsample_block.has_cross_attention
                ):
                    sample = upsample_block(
                        hidden_states=sample,
                        res_hidden_states_tuple=res_samples,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        image_only_indicator=image_only_indicator,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        res_hidden_states_tuple=res_samples,
                        temb=emb,
                        image_only_indicator=image_only_indicator,
                    )
                _to_offload(upsample_block)

                # Free loaded residuals
                if use_residual_cache:
                    del res_samples
                    if main_device.type == "cuda":
                        torch.cuda.empty_cache()

        # Clean up residual cache
        if use_residual_cache:
            residual_cache.cleanup()

        # 6. post-process
        _to_main(self.conv_norm_out)
        _to_main(self.conv_out)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        _to_offload(self.conv_norm_out)
        _to_offload(self.conv_out)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
