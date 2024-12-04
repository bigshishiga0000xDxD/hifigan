import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms, spectrogram_transform = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = {
        k: v.to(device)
        for k, v in instantiate(config.loss_function).items()
    }
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    optimizer_G = instantiate(config.optimizer, params=model.generator.parameters())
    optimizer_D = instantiate(config.optimizer, params=model.discriminators.parameters())
    optimizer = {"generator": optimizer_G, "discriminator": optimizer_D}

    # dummy_variable = torch.nn.Parameter(torch.zeros(1))
    dummy_optimizer = instantiate(config.optimizer, params=[torch.empty(1)])
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=dummy_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        spectrogram_transform=spectrogram_transform,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
