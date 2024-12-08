import os
from pathlib import Path

from src.datasets import BaseDataset
from src.utils.io_utils import read_json, write_json


class InlineDataset(BaseDataset):
    def __init__(self, transcription: str, *args, **kwargs):
        index = [{"normalized_transcript": transcription}]
        super().__init__(index, *args, **kwargs)
