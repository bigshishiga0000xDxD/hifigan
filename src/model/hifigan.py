from torch import nn
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
import torch

from src.model.base_model import BaseModel
from src.transforms.spectrogram import MelSpectrogramConfig

mel_spectrogram_config = MelSpectrogramConfig()


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: list[list[int]],
        slope: float
    ):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(slope)
        self.convs = nn.ModuleList([
            nn.ModuleList([
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2))
                for dilation in dr
            ])
            for dr in dilations
        ])
    
    def forward(self, x: torch.Tensor):
        for convs in self.convs:
            residual = x
            for conv in convs:
                x = self.leaky_relu(x)
                x = conv(x)
            x = residual + x

        return x

class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int,
        channels: list[int],
        kernel_size: int,
        stride: int,
        slope: float
    ):
        super().__init__()

        self.period = period
        self.leaky_relu = nn.LeakyReLU(slope)

        channels = [1] + channels
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(
                in_channels,
                out_channels,
                (kernel_size, 1),
                (stride, 1),
                padding=(kernel_size // 2, 0)
            ))
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ])

        self.collapse = weight_norm(nn.Conv2d(
            channels[-1], 1, kernel_size=(3, 1), padding=(1, 0)
        ))
    
    def forward(self, x: torch.Tensor):
        b, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = nn.functional.pad(x, (0, n_pad), mode="reflect")
        x = x.view(b, 1, -1, self.period)

        activations = []
        for conv in self.convs:
            x = conv(x)
            x = self.leaky_relu(x)
            activations.append(x)
        x = self.collapse(x)
        activations.append(x)

        x = x.view(b, -1).mean(dim=1)
        return x, activations


class ScaleDiscriminator(nn.Module):
    def __init__(
        self,
        scale: int,
        channels: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        groups_sizes: list[int],
        slope: float,
        norm: type[weight_norm] | type[spectral_norm]
    ):
        super().__init__()

        self.leaky_relu = nn.LeakyReLU(slope)
        if scale == 1:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AvgPool1d(kernel_size=2 * scale, stride=scale, padding=scale)

        channels = [1] + channels
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups
            ))
            for in_channels, out_channels, kernel_size, stride, groups in zip(
                channels[:-1], channels[1:], kernel_sizes, strides, groups_sizes
            )
        ])

        self.collapse = norm(nn.Conv1d(
            channels[-1], 1, kernel_size=3, padding=1
        ))
    
    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = x.unsqueeze(1)

        activations = []
        for conv in self.convs:
            x = conv(x)
            x = self.leaky_relu(x)
            activations.append(x)
        x = self.collapse(x)
        activations.append(x)

        x = x.view(x.shape[0], -1).mean(dim=1)
        return x, activations

class Generator(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        upsample_kernel_sizes: list[int],
        residual_kernel_sizes: list[int],
        residual_dilations: list[list[int]],
        slope: float
    ):
        super().__init__()
        self.hop_length = mel_spectrogram_config.hop_length

        self.leaky_relu = nn.LeakyReLU(slope)
        self.expand = weight_norm(nn.Conv1d(
            mel_spectrogram_config.n_mels,
            hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
        ))

        self.upsamples = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(
                hidden_channels // 2**i,
                hidden_channels // 2**(i + 1),
                kernel_size=upsample_kernel_sizes[i], 
                stride=upsample_kernel_sizes[i] // 2,
                padding=upsample_kernel_sizes[i] // 4
            ))
            for i in range(len(upsample_kernel_sizes))
        ])

        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(
                    hidden_channels // 2**(i + 1),
                    residual_kernel_sizes[j],
                    residual_dilations,
                    slope
                ) 
                for j in range(len(residual_kernel_sizes))
            ])
            for i in range(len(upsample_kernel_sizes))
        ])

        self.collapse = weight_norm(nn.Conv1d(
            hidden_channels // 2**len(upsample_kernel_sizes),
            1,
            kernel_size=7,
            stride=1,
            padding=3,
        ))
    
    def forward(self, spec: torch.Tensor, spec_length: torch.Tensor, **batch):
        x = self.expand(spec)

        for upsample, res_blocks in zip(self.upsamples, self.res_blocks):
            x = self.leaky_relu(x)
            x = upsample(x)

            x_sum = 0
            for res_block in res_blocks:
                x_sum += res_block(x)
            x = x_sum / len(res_blocks)

        x = self.leaky_relu(x)
        x = self.collapse(x)
        x = torch.tanh(x)

        return {
            'output': x.squeeze(1),
            'output_length': self._transform_lengths(spec_length)
        }
    
    def _transform_lengths(self, spec_length: torch.Tensor) -> torch.Tensor:
        output_len = spec_length.clone()
        for upsample in self.upsamples:
            output_len *= upsample.stride[0]
        return output_len


class HiFiGAN(BaseModel):
    def __init__(
        self,
        g_hidden_channels: int,
        g_upsample_kernel_sizes: list[int],
        g_residual_kernel_sizes: list[int],
        g_residual_dilations: list[list[int]],
        mpd_periods: list[int],
        mpd_channels: list[int],
        mpd_kernel_size: int,
        mpd_stride: int,
        msd_scales: list[int],
        msd_channels: list[int],
        msd_kernel_sizes: list[int],
        msd_strides: list[int],
        msd_groups_sizes: list[int],
        msd_norms: list[type[weight_norm] | type[spectral_norm]],
        slope: float = 0.1
    ):
        super().__init__()

        self.generator = Generator(
            g_hidden_channels,
            g_upsample_kernel_sizes,
            g_residual_kernel_sizes,
            g_residual_dilations,
            slope=slope
        )

        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(
                period,
                mpd_channels,
                mpd_kernel_size,
                mpd_stride,
                slope=slope
            )
            for period in mpd_periods
        ])

        self.discriminators += [
            ScaleDiscriminator(
                scale,
                msd_channels,
                msd_kernel_sizes,
                msd_strides,
                msd_groups_sizes,
                slope,
                norm
            )
            for scale, norm in zip(msd_scales, msd_norms)
        ]

    def generate(self, **batch):
        return self.generator(**batch)
    
    forward = generate

    def discriminate(self, wav, output, **batch):
        real_activations, fake_activations, real_scores, fake_scores = [], [], [], []

        for d in self.discriminators:
            (r_score, r_act), (f_score, f_act) = d(wav), d(output)
            real_scores.append(r_score)
            fake_scores.append(f_score)
            real_activations.append(r_act)
            fake_activations.append(f_act)

        return {
            'real_scores': real_scores,
            'fake_scores': fake_scores,
            'real_activations': real_activations,
            'fake_activations': fake_activations
        }