import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate


def collate_fn(dataset_items: list[dict], unpad: bool, max_length: int | None):
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

    if "wav" in dataset_items[0]:
        result_batch["wav"] = pad_sequence(
            [item["wav"] for item in dataset_items], batch_first=True, padding_value=0
        )

    result_batch.update(
        default_collate(
            [
                {key: value for key, value in item.items() if key != "wav"}
                for item in dataset_items
            ]
        )
    )

    if unpad:
        # we cut all wavs to the length of the shortest one
        # to avoid paddings (for the maximum efficiency)
        min_length = torch.min(result_batch["wav_length"])
        min_length = torch.minimum(min_length, torch.tensor(max_length))

        cut_wavs = []
        for sample, length in zip(result_batch["wav"], result_batch["wav_length"]):
            # [l, r)
            offset = torch.randint(0, length - min_length + 1, (1,))
            # sample random segment of length min_length
            cut_wavs.append(sample[offset : offset + min_length])

        result_batch["wav"] = torch.stack(cut_wavs)
        result_batch["wav_length"] = min_length.repeat(len(dataset_items))

    return result_batch


def get_collate_fn(unpad: bool, max_length: int | None = None):
    return lambda dataset_items: collate_fn(dataset_items, unpad, max_length)
