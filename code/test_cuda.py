"""
This script checks the availability of CUDA (NVIDIA's parallel computing platform
and application programming interface)
for PyTorch and provides information about the GPU if available.

If CUDA is available, it prints the following information:
- A message indicating that CUDA is available and that a GPU is present.
- The count of available CUDA devices.
- The index of the current CUDA device being used.
- The name of the current CUDA device.

If CUDA is not available, it prints a message indicating that the script is running on the CPU.

Usage:
- This script is useful for verifying the GPU setup in a PyTorch environment
and ensuring that the necessary hardware is available for running deep learning models.
"""

import torch

if torch.cuda.is_available():
    print("CUDA is available! You have a GPU.")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is NOT available. Running on CPU.")
