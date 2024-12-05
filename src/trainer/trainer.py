import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms import MelSpectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        batch.update(self.spectrogram_transform(**batch))

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # Generator update
        batch.update(self.model.generate(**batch))

        self.transform = MelSpectrogram().to(self.device)
        inv_spec = self.transform(
            wav=batch["output"], wav_length=batch["output_length"]
        )
        batch["output_spec"] = inv_spec["spec"]
        batch["output_spec_length"] = inv_spec["spec_length"]

        batch.update(self.model.discriminate(**batch))
        batch.update(self.criterion["generator"](**batch))

        if self.is_train:
            self.optimizer["generator"].zero_grad()
            batch["generator_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["generator"].step()

        # Discriminator update
        batch["output"] = batch["output"].detach()
        batch.update(self.model.discriminate(**batch))
        batch.update(self.criterion["discriminator"](**batch))

        if self.is_train:
            self.optimizer["discriminator"].zero_grad()
            batch["discriminator_loss"].backward()
            self._clip_grad_norm()
            self.optimizer["discriminator"].step()

        if self.is_train:
            self._update_lr()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """

        self._log_audio("input_wav", batch["wav"], batch["wav_length"])
        self._log_audio("output_wav", batch["output"], batch["output_length"])
        self._log_spectrogram("input_spectrogram", batch["spec"], batch["spec_length"])
        self._log_spectrogram(
            "output_spectrogram", batch["output_spec"], batch["output_spec_length"]
        )

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass

    def _log_spectrogram(self, name, spec, spec_length):
        spectrogram_for_plot = spec[0, :, : spec_length[0]].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(name, image)

    def _log_audio(self, name, audio, length):
        self.writer.add_audio(name, audio[0, : length[0]], sample_rate=self.sample_rate)
