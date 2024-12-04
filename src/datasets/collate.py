import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}
    result_batch["wav"] = pad_sequence(
        [item["wav"] for item in dataset_items], batch_first=True, padding_value=0
    )

    result_batch.update(default_collate([
        {key: value for key, value in item.items() if key != "wav"}
        for item in dataset_items
    ]))

    return result_batch
