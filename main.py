import torch
import sys


def system_check():
    print(f"--- Python Version: {sys.version.split()[0]} ---")
    print(f"--- PyTorch Version: {torch.__version__} ---")

    if torch.cuda.is_available():
        print(f"CUDA is available! Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available (Running on CPU)")


if __name__ == "__main__":
    print("Starting PyTorch Container Environment...")
    system_check()