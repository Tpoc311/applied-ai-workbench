import os

import torch
import torchvision


def main():
    print(f"pytorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"User: {os.getlogin()}")


if __name__ == "__main__":
    main()
