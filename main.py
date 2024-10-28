import os
import sys
from pathlib import Path


def main():
    """note: 58epochs performs better than the 218epochs"""
    script_path = "recipes/dns_interspeech_2020/inference.py"

    config_path = "recipes/dns_interspeech_2020/fullsubnet/inference.toml"
    ckpt_path = "ckpt/fullsubnet_best_model_58epochs.tar"

    # config_path = "recipes/dns_interspeech_2020/fullsubnet/inference_cum.toml"
    # ckpt_path = "ckpt/cum_fullsubnet_best_model_218epochs.tar"

    in_dirs = [
        # "/home/featurize/data/audio_test/orig_dataset/LibriSpeech/train-clean-100",
        # "/home/featurize/data/audio_test/orig_dataset/LibriSpeech/dev-clean",
        # "/home/featurize/data/audio_test/orig_dataset/LibriSpeech/test-clean",
        "/home/featurize/data/audio_test/orig_dataset/aishell3",
    ]
    out_base_dir = Path("/home/featurize/data/audio_test/denoised/aishell3")

    for in_dir in in_dirs:
        out_dir = out_base_dir / Path(in_dir).name
        cmd = f"{sys.executable} {script_path} -C {config_path} -M {ckpt_path} -I {in_dir} -O {out_dir}"
        os.system(cmd)
    ...


if __name__ == "__main__":
    main()
    ...
