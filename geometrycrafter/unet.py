from typing import Union, Tuple

import torch
from diffusers import UNetSpatioTemporalConditionModel
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionOutput
from diffusers.utils import is_torch_version


class UNetSpatioTemporalConditionModelVid2vid(
    UNetSpatioTemporalConditionModel
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._block_cpu_offload = False
        self._main_device = None
        self._offload_device = torch.device("cpu")

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
        block_offload = self._block_cpu_offload
        main_device = self._main_device if self._main_device is not None else sample.device
        offload_device = self._offload_device if hasattr(self, "_offload_device") else torch.device("cpu")

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
