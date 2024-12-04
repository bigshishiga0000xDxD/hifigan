import torch
from torch import nn

from src.transforms import MelSpectrogram


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        real_scores: list[torch.Tensor],
        fake_scores: list[torch.Tensor],
        real_labels: torch.Tensor,
        fake_labels: torch.Tensor,
        **batch
    ) -> float:
        loss = 0

        # iterating over different discriminators
        for real, fake in zip(real_scores, fake_scores):
            real_loss = self.loss_fn(real, real_labels)
            fake_loss = self.loss_fn(fake, fake_labels)
            loss += real_loss + fake_loss

        return loss

class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss()
    
    def forward(
        self,
        spec: torch.Tensor,
        spec_length: torch.Tensor,
        output_spec: torch.Tensor,
        output_spec_length: torch.Tensor,
        **batch
    ) -> float:
        if not torch.all(output_spec_length == spec_length):
            idx = torch.nonzero(output_spec_length != spec_length)[0].item()
            raise RuntimeError(
                f"Difference in the spectrogram lengths: {output_spec_length[idx].item()} vs "
                f"{spec_length[idx].item()}."
            )

        # since padding masks are the same, taking loss over all items should be fine
        return self.loss_fn(output_spec, spec)

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_fn = nn.L1Loss()
    
    def forward(
        self,
        real_activations: list[list[torch.Tensor]],
        fake_activations: list[list[torch.Tensor]],
        **batch
    ):
        # activations[i][j] is the j-th activation of discriminator i of shape (B x F)
        # where B is the batch size and F is the feature dimension 
        # according to the paper, we sum over i and j and average over B and F

        loss = 0
        for real, fake in zip(real_activations, fake_activations):
            for r, f in zip(real, fake):
                r = r[..., :f.shape[-1]]
                loss += self.loss_fn(r, f)

        return loss

        

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_mel: float = 45, lambda_fm: float = 2):
        super().__init__()
        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm

        self.adv_loss = AdversarialLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.fm_loss = FeatureMatchingLoss()
    
    def forward(
        self,
        **batch
    ) -> dict:
        adv_loss = self.adv_loss(
            **batch,
            real_labels=torch.zeros_like(batch['real_scores'][0]),
            fake_labels=torch.ones_like(batch['fake_scores'][0])
        )

        mel_loss = self.mel_loss(**batch) * self.lambda_mel

        fm_loss = self.fm_loss(**batch) * self.lambda_fm

        return {
            'generator_loss': adv_loss + mel_loss + fm_loss,
            'g_adv_loss': adv_loss,
            'g_mel_loss': mel_loss
        }

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.adv_loss = AdversarialLoss()
    
    def forward(
        self,
        **batch
    ) -> dict:
        adv_loss = self.adv_loss(
            **batch,
            real_labels=torch.ones_like(batch['real_scores'][0]),
            fake_labels=torch.zeros_like(batch['fake_scores'][0])
        )

        return {
            'discriminator_loss': adv_loss,
            'd_adv_loss': adv_loss
        }