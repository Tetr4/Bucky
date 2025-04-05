

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class CudaDevice:
    index: int

    @property
    def name(self) -> str:
        return torch.cuda.get_device_name(self.index)

    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda", self.index)

    @property
    def free_memory(self) -> int:
        return torch.cuda.mem_get_info(self.index)[0]

    @property
    def total_memory(self) -> int:
        return torch.cuda.mem_get_info(self.index)[1]

    def __str__(self) -> str:
        return f"Device[{self.index}]: {self.name}, Total Memory: {self.total_memory / (1024**3):.1f} GB, Free Memory: {self.free_memory / (1024**3):.1f} GB"


def get_cuda_devices() -> list[CudaDevice]:
    return [CudaDevice(i) for i in range(torch.cuda.device_count())]


def get_free_cuda_device(free_memory: int = 0) -> Optional[CudaDevice]:
    devs = sorted((dev for dev in get_cuda_devices() if dev.free_memory >= free_memory), key=lambda dev: dev.free_memory, reverse=True)
    return next(iter(devs), None)
