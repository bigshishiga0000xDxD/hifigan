import nltk
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.transforms.spectrogram import MelSpectrogramConfig

nltk.download("averaged_perceptron_tagger_eng")
config = MelSpectrogramConfig()


class FastSpeech2Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        model, cfg, task = load_model_ensemble_and_task_from_hf_hub(
            "facebook/fastspeech2-en-ljspeech",
            arg_overrides={"vocoder": "hifigan", "fp16": False}
            # it doesn't actually run hifigan
        )
        model = model[0].to(device)
        generator = task.build_generator([model], cfg)
        generator.vocoder.to(device)

        self.task = task
        self.device = device
        self.model = model
        self.generator = generator

    def forward(self, normalized_transcript: list[str], **batch):
        result = []
        lenghts = []
        for text in normalized_transcript:
            sample = TTSHubInterface.get_model_input(self.task, text)
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(
                self.device
            )
            spec = self.generator.generate(self.model, sample)[0]["feature"]

            result.append(spec)
            lenghts.append(spec.shape[0])

        return {
            "spec": pad_sequence(
                result, batch_first=True, padding_value=config.pad_value
            )
            .transpose(1, 2)
            .contiguous(),
            "spec_length": torch.tensor(lenghts).to(self.device),
        }
