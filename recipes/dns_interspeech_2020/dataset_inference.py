from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import basename


class Dataset(BaseDataset):
    def __init__(
        self,
        dataset_dir_list,
        sr,
    ):
        """Inference dataset, which only contains noisy files.

        Args:
            noisy_dataset_dir_list (str or list): noisy dir or noisy dir list
        """
        super().__init__()
        assert isinstance(dataset_dir_list, list)
        self.sr = sr

        noisy_file_info_list: List[Tuple[str, str]] = []
        for dataset_dir in dataset_dir_list:
            dataset_dir = Path(dataset_dir).expanduser().absolute().as_posix()
            noisy_file_path_list = librosa.util.find_files(dataset_dir)  # Sorted
            noisy_file_info_list += list(map(lambda x: (dataset_dir, x), noisy_file_path_list))

        self.noisy_file_info_list = noisy_file_info_list
        self.length = len(self.noisy_file_info_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        dataset_dir, noisy_file_path = self.noisy_file_info_list[item]
        noisy_y = librosa.load(noisy_file_path, sr=self.sr)[0]
        noisy_y = noisy_y.astype(np.float32)

        return noisy_y, dataset_dir, noisy_file_path
