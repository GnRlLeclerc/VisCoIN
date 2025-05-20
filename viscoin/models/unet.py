import torch
from diffusers import UNet2DConditionModel


class DiffusionUNetAdapted(torch.nn.Module):

    def __init__(self, unet: UNet2DConditionModel):
        super(DiffusionUNetAdapted, self).__init__()

        self.unet = unet

        self.original_sample_shape = None

    def downwards(
        self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """
        Downwards pass through the UNet.
        See UNet2DConditionModel for more details.
        Returns the downward blocks features and the time embedding.
        """

        self.original_sample_shape = sample.shape[2:]

        if self.unet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # Compute the embeddings
        t_emb = self.unet.get_time_embed(sample=sample, timestep=timestep)
        emb = self.unet.time_embedding(t_emb, None)

        aug_emb = self.unet.get_aug_embed(
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=None,
        )

        emb = emb + aug_emb if aug_emb is not None else emb

        encoder_hidden_states = self.unet.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=None
        )

        # Preprocess the encoder hidden states if necessary
        sample = self.unet.conv_in(sample)

        # Downwards pass
        down_block_res_samples = (sample,)
        for downsample_block in self.unet.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        return sample, down_block_res_samples, emb, encoder_hidden_states

    def middle(self, sample: torch.Tensor, emb: torch.Tensor, encoder_hidden_states: torch.Tensor):
        """
        Middle block pass of the UNet.
        Returns the middle block features.
        """
        if self.unet.mid_block is not None:
            if (
                hasattr(self.unet.mid_block, "has_cross_attention")
                and self.unet.mid_block.has_cross_attention
            ):
                sample = self.unet.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = self.unet.mid_block(sample, emb)

        return sample

    def upwards(
        self,
        sample: torch.Tensor,
        down_block_res_samples: tuple[torch.Tensor, ...],
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ):
        """
        Upwards pass through the UNet.
        Returns the noise residuals.
        """

        default_overall_up_factor = 2**self.unet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in self.original_sample_shape:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        for i, upsample_block in enumerate(self.unet.up_blocks):
            is_final_block = i == len(self.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)

        return sample

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor
    ):
        """
        Forward pass through the UNet.
        Returns the noise residuals.
        """
        sample, down_block_res_samples, emb, encoder_hidden_states = self.downwards(
            sample, timestep, encoder_hidden_states
        )
        sample = self.middle(sample, emb, encoder_hidden_states)
        sample = self.upwards(sample, down_block_res_samples, emb, encoder_hidden_states)

        return sample

    def sample(self, num_samples: int, seed: int = None):
        """
        Samples random noise in the dimensions of the Unet
        Taken from semantic-diffusion : https://github.com/renhaa/semantic-diffusion/blob/main/semanticdiffusion.py
        """
        if seed is None:
            seed = torch.randint(int(1e6), (1,))

        return torch.randn(
            num_samples,
            self.unet.in_channels,
            self.unet.sample_size,
            self.unet.sample_size,
            generator=torch.manual_seed(seed),
        ).to(self.unet.device)
