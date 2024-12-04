from dataclasses import dataclass

import librosa
import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):
    def __init__(self, config: MelSpectrogramConfig = MelSpectrogramConfig()):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max,
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    @torch.no_grad()
    def forward(
        self, wav: torch.Tensor, wav_length: torch.Tensor, **batch
    ) -> torch.Tensor:
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(0)
        # assuming B x T now

        # can't use batched forward since we have different lengths (i think)
        specs = []
        for sample, length in zip(wav, wav_length):
            sample = sample[:length]

            mel = self.mel_spectrogram(sample).clamp_(min=1e-5).log_()
            specs.append(mel)

        mel = pad_sequence(
            [
                # spec is n_mel x L, `pad_sequence` requires L x ...
                spec.transpose(0, -1)
                for spec in specs
            ],
            batch_first=True,
            padding_value=self.config.pad_value,
        ).transpose(1, -1)

        return {
            "spec": mel[..., :-1],
            "spec_length": wav_length // self.config.hop_length,
        }
