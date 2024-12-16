import random
import numpy as np
from pathlib import Path
import librosa
import torch
import soundfile as sf
import os
from tqdm import tqdm
import multiprocessing

from audio_zen.acoustics.feature import (
    is_clipped,
    load_wav,
    norm_amplitude,
    tailor_dB_FS,
)


class AddNoise(object):
    def __init__(
        self,
        clean_dir,
        noise_dir,
        snr_range,
        silence_length,
        target_dB_FS,
        target_dB_FS_floating_value,
        sub_sample_length,
        sr,
    ):
        """Dynamic generate mixing data for training"""
        super().__init__()

        # 干净数据读取
        clean_files_list = []
        clean_dir = Path(clean_dir).expanduser().absolute()
        clean_files_list += librosa.util.find_files((clean_dir / "clean").as_posix())

        # 噪声数据读取
        noise_files_list = []
        noise_dir = Path(noise_dir).expanduser().absolute()
        noise_files_list += librosa.util.find_files(noise_dir.as_posix())

        self.clean_list = clean_files_list
        self.noise_list = noise_files_list

        # 根据输入的信噪比区间产生整数的信噪比
        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length
        self.sr = sr

        self.length = len(self.clean_list)

    def __len__(self):
        return self.length

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(int(self.sr * self.silence_length), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_list)
            noise_new_added = load_wav(noise_file, sr=self.sr)
            noise_new_added = noise_new_added[:len(noise_new_added) // 3]  # 只取长度的1/3
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # If still need to add new noise, insert a small silence segment firstly
            if remaining_length > 0:
                silence_len = min(remaining_length, len(silence))
                noise_y = np.append(noise_y, silence[:silence_len])
                remaining_length -= silence_len

        if len(noise_y) > target_length:
            idx_start = np.random.randint(len(noise_y) - target_length)
            noise_y = noise_y[idx_start: idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(
        clean_y,
        noise_y,
        snr,
        target_dB_FS,
        target_dB_FS_floating_value,
        eps=1e-6,
    ):
        """Mix clean_y and noise_y based on a given SNR and a RIR (if exist).

        Args:
            clean_y: clean signal
            noise_y: noise signal
            snr (int): signal-to-noise ratio
            target_dB_FS (int): target dBFS
            target_dB_FS_floating_value (int): target dBFS floating value
            eps: eps

        Returns:
            (noisy_y, clean_y)
        """
        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y ** 2).mean() ** 0.5

        snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value,
        )

        # Use the same RMS value of dBFS for noisy speech
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # The mixed speech is clipped if the RMS value of noisy speech is too large.
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    @staticmethod
    def _parse_snr_range(snr_range):
        assert (
            len(snr_range) == 2
        ), f"The range of SNR should be [low, high], not {snr_range}."
        assert (
            snr_range[0] <= snr_range[-1]
        ), f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list

    def add_noise_to_file(self, clean):
        clean_y = load_wav(clean, sr=self.sr)
        noise_y = self._select_noise_y(target_length=len(clean_y))
        assert len(clean_y) == len(noise_y), f"Inequality: {len(clean_y)=} {len(noise_y)=}"
        snr = self._random_select_from(self.snr_list)
        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value
        )

        # 创建路径
        noisy_path = clean.replace("clean", "noisy")
        os.makedirs(os.path.dirname(noisy_path), exist_ok=True)
        sf.write(noisy_path, noisy_y, self.sr)

    def add_noise_to_files(self):
        with multiprocessing.Pool() as pool:
            list(tqdm(pool.imap(self.add_noise_to_file, self.clean_list), total=self.length))


if __name__ == "__main__":
    torch.manual_seed(2023)
    np.random.seed(2023)
    random.seed(2023)
    add_noise = AddNoise(
        clean_dir="/*/Noisy_Rain",
        noise_dir="/*/Noise",
        snr_range=[10, 10],
        silence_length=0.1,
        target_dB_FS=25,
        target_dB_FS_floating_value=10,
        sub_sample_length=3,
        sr=32000,
    )
    add_noise.add_noise_to_files()
