import os
import shutil
import sys
from pathlib import Path


def main():
    """note: 58epochs performs better than the 218epochs"""
    script_path = "recipes/dns_interspeech_2020/inference.py"

    config_path = "recipes/dns_interspeech_2020/fullsubnet/inference.toml"
    ckpt_path = "ckpt/fullsubnet_best_model_58epochs.tar"

    # config_path = "recipes/dns_interspeech_2020/fullsubnet/inference_cum.toml"
    # ckpt_path = "ckpt/cum_fullsubnet_best_model_218epochs.tar"

    in_dir = r"D:\Temp\picked_files"
    out_dir = r"D:\Temp\denoised_files"
    cmd = f"{sys.executable} {script_path} -C {config_path} -M {ckpt_path} -I {in_dir} -O {out_dir}"

    # out_dir = Path(out_dir)
    # shutil.rmtree(out_dir)
    # out_dir.mkdir()

    os.system(cmd)
    ...


if __name__ == '__main__':
    main()
    ...
