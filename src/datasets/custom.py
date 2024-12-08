import os
from pathlib import Path

from src.datasets import BaseDataset
from src.utils.io_utils import read_json, write_json


class CustomDirDataset(BaseDataset):
    def __init__(self, root: str, *args, **kwargs):
        self.root = Path(root)

        index_path = self._get_index_path()
        if not index_path.exists():
            self._build_index()

        index = read_json(index_path)
        super().__init__(index, *args, **kwargs)

    def _build_index(self) -> None:
        has_text = os.path.exists(self.root / "transcriptions")
        has_audio = os.path.exists(self.root / "wavs")
        assert (
            has_text ^ has_audio
        ), "Ambiguous dataset structure. Provide either text or audio."

        mode = "text" if has_text else "audio"

        if mode == "text":
            path = self.root / "transcriptions"
            suffix = ".txt"
            field = "txt_path"
        else:
            path = self.root / "wavs"
            suffix = (".wav", ".flac", ".mp3", ".ogg")
            field = "wav_path"

        index = []
        for name in filter(lambda x: x.endswith(suffix), os.listdir(path)):
            obj = str(path / name)
            index.append({field: obj})

        write_json(index, self._get_index_path())

    def _get_index_path(self) -> Path:
        return self.root / "index.json"
