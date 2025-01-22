import gc
import logging
from typing import List, TypeVar
import subprocess
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
T = TypeVar("T")

def get_free_gpus():
    """
    Identifies GPUs that have no active processes using nvidia-smi pmon.
    Returns:
        List of GPU IDs with no active processes.
    """
    try:
        # Run `nvidia-smi pmon` to get process stats
        pmon_output = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],  # Run only once (-c 1)
            stdout=subprocess.PIPE,
            text=True
        ).stdout

        # Parse the output
        lines = pmon_output.strip().split("\n")
        
        # Extract GPU IDs with active processes
        free_gpu_ids = set()
        for line in lines:
            if line.startswith("#") or not line.strip():  # Skip header and empty lines
                continue
            columns = line.split()
            gpu_id = columns[0]   # GPU ID is the first column
            if columns[1] == '-':  # No active processes
                free_gpu_ids.add(int(gpu_id))

    except Exception as e:
        print(f"Error: {e}")
        return []

    return list(free_gpu_ids)

_device = None
def get_torch_device(device: str = "auto") -> str:
    global _device
    if _device is None:
        if device == "auto":
            if torch.cuda.is_available() and len(get_free_gpus()) > 0:
                free_gpus_list = get_free_gpus()
                _device = f"cuda:{free_gpus_list[0]}"
            elif torch.backends.mps.is_available():
                _device = "mps"
            else:
                _device = "cpu"
            logger.info(f"Using device: {_device}")
        else:
            _device = device

    return _device


def tear_down_torch():
    """
    Teardown for PyTorch.
    Clears GPU cache for the specific device in use and resets `_device` to None.
    """
    global _device
    gc.collect()

    if _device is None:
        logger.warning("Device is not initialized. No teardown needed.")
        return

    if "cuda" in _device and torch.cuda.is_available():
        torch.cuda.set_device(_device)
        torch.cuda.empty_cache()
        logger.info(f"Cleared CUDA cache for device {_device}.")
    elif "mps" in _device and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache.")
    else:
        logger.info("No GPU or MPS device to clear.")



class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]
