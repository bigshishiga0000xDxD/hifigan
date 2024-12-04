from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio

from src.datasets import BaseDataset
from src.utils.io_utils import read_json, write_json


class LJSpeechDataset(BaseDataset):
    def __init__(
        self, root: str, part: str, train_split: float | None = None, *args, **kwargs
    ):
        self.root = Path(root)
        _ = torchaudio.datasets.LJSPEECH(self.root, download=True)
        self.root = self.root / "LJSpeech-1.1"

        index_path = self._get_index_path(part)
        if not index_path.exists():
            assert train_split is not None
            self._build_index(train_split)

        index = read_json(index_path)
        super().__init__(index, *args, **kwargs)

    def _build_index(self, train_split: float) -> None:
        index = []
        metadata = pd.read_csv(self.root / "metadata.csv", header=None, sep="|")
        for _, row in metadata.iterrows():
            transcript = row[2]

            # skip weird symbols
            if isinstance(transcript, str) and transcript and transcript.isascii():
                index.append(
                    {
                        "wav_path": str(self.root / "wavs" / f"{row[0]}.wav"),
                        "normalized_transcript": transcript,
                    }
                )

        train_index, val_index = self._random_split(index, train_split)
        write_json(train_index, self._get_index_path("train"))
        write_json(val_index, self._get_index_path("val"))

    def _get_index_path(self, part: str) -> Path:
        return self.root / f"{part}_index.json"

    @staticmethod
    def _random_split(
        index: list[dict], train_split: float
    ) -> tuple[list[dict], list[dict]]:
        perm = np.random.permutation(len(index))
        train_size = int(len(index) * train_split)
        return (
            np.take(index, perm[:train_size]).tolist(),
            np.take(index, perm[train_size:]).tolist(),
        )
