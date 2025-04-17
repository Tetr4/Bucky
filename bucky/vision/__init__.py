from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CameraStream(ABC, object):
    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError()
