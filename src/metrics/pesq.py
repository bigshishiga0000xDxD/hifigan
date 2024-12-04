import torch
import numpy as np
import torchaudio.transforms as T

from src.metrics.base_metric import BaseMetric


class PESQMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        self.metric = metric.to(device)
        self.resample = T.Resample(22050, 8000).to(device)

    @torch.inference_mode()
    def __call__(
        self,
        wav: torch.Tensor,
        output: torch.Tensor,
        wav_length: torch.Tensor,
        output_length: torch.Tensor,
        **batch
    ) -> float:
        assert torch.all(wav_length == output_length)

        values = []
        for output_sample, input_sample, length in zip(output, wav, wav_length):
            output_sample = self.resample(output_sample[:length])
            input_sample = self.resample(input_sample[:length])
            values.append(self.metric(output_sample, input_sample).item())

        return np.mean(values)