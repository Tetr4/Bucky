from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar
import numpy as np
import torch

T = TypeVar('T')


class SpeechDenoiser(ABC, Generic[T]):
    def __init__(self,
                 chunk_buffer_size: int = 128,
                 min_chunk_offset: int = 8,
                 max_chunk_offset: int = 10):
        assert min_chunk_offset <= max_chunk_offset
        assert max_chunk_offset <= chunk_buffer_size

        self.min_chunk_offset = min_chunk_offset
        self.max_chunk_offset = max_chunk_offset
        self.next_chunk_offset = max_chunk_offset

        self.chunk_buffer_size = chunk_buffer_size
        self.chunk_buffer: list[T] = []

        self.last_result: Optional[T] = None

    @abstractmethod
    def _convert_samples(self, samples: bytes) -> T:
        raise NotImplementedError()

    @abstractmethod
    def _denoise_chunk_buffer(self, chunk_buffer: list[T], sample_rate: int) -> T:
        raise NotImplementedError()

    @abstractmethod
    def _get_denoised_slice(self, denoised_samples: T, start_idx: int, end_idx: Optional[int]) -> bytes:
        raise NotImplementedError()

    def denoise(self, samples: bytes, sample_width: int, sample_rate: int) -> bytes:
        assert sample_width == 2

        noisy_chunk: T = self._convert_samples(samples)
        self.chunk_buffer.append(noisy_chunk)

        if len(self.chunk_buffer) > self.chunk_buffer_size:
            self.chunk_buffer.pop(0)

        # make sure that the buffer is always full
        while len(self.chunk_buffer) < self.chunk_buffer_size:
            self.chunk_buffer.append(noisy_chunk)

        if self.last_result is None:
            self.last_result = self._denoise_chunk_buffer(self.chunk_buffer, sample_rate)
            self.next_chunk_offset = self.max_chunk_offset

        start_idx: int = -len(noisy_chunk) * min(self.next_chunk_offset, len(self.chunk_buffer))
        end_idx: int = start_idx + len(noisy_chunk)
        clean_chunk: T = self._get_denoised_slice(self.last_result, start_idx, end_idx if end_idx else None)

        self.next_chunk_offset -= 1
        if self.next_chunk_offset < self.min_chunk_offset:
            self.last_result = None

        return clean_chunk


class SpeechDenoiserNR(SpeechDenoiser[np.ndarray]):
    def __init__(self,
                 chunk_buffer_size: int = 128,
                 min_chunk_offset: int = 8,
                 max_chunk_offset: int = 10):
        super().__init__(chunk_buffer_size=chunk_buffer_size,
                         min_chunk_offset=min_chunk_offset,
                         max_chunk_offset=max_chunk_offset)

    def _convert_samples(self, samples: bytes) -> np.ndarray:
        return np.frombuffer(samples, dtype=np.int16)

    def _denoise_chunk_buffer(self, chunk_buffer: list[np.ndarray], sample_rate: int) -> np.ndarray:
        import noisereduce as nr
        noisy_data: np.ndarray = np.concatenate(chunk_buffer)
        return nr.reduce_noise(y=noisy_data,
                               sr=sample_rate,
                               stationary=True,
                               use_torch=True,
                               n_fft=min(1024, len(noisy_data)))

    def _get_denoised_slice(self, denoised_samples: np.ndarray, start_idx: int, end_idx: Optional[int]) -> bytes:
        return denoised_samples[start_idx:end_idx].astype(np.int16).tobytes()


class SpeechDenoiserDF(SpeechDenoiser[torch.Tensor]):
    def __init__(self,
                 chunk_buffer_size: int = 128,
                 min_chunk_offset: int = 8,
                 max_chunk_offset: int = 10):
        super().__init__(chunk_buffer_size=chunk_buffer_size,
                         min_chunk_offset=min_chunk_offset,
                         max_chunk_offset=max_chunk_offset)

        from df.enhance import init_df
        self.model, self.df_state, _ = init_df(log_level="none", post_filter=False)

    def _convert_samples(self, samples: bytes) -> torch.Tensor:
        return torch.frombuffer(samples, dtype=torch.int16).to(torch.float32) / (1 << 15)

    def _denoise_chunk_buffer(self, chunk_buffer: list[np.ndarray], sample_rate: int) -> torch.Tensor:
        from torchaudio.functional import resample
        from df.enhance import enhance
        full_input_tensor: torch.Tensor = torch.cat(chunk_buffer, dim=0)
        input_tensor: torch.Tensor = full_input_tensor.unsqueeze(0)

        if self.df_state.sr() != sample_rate:
            input_tensor = resample(input_tensor, sample_rate, self.df_state.sr(),
                                    resampling_method="sinc_interp_hann", lowpass_filter_width=16)

        enhanced_tensor: torch.Tensor = enhance(self.model, self.df_state, input_tensor)

        if self.df_state.sr() != sample_rate:
            enhanced_tensor = resample(enhanced_tensor, self.df_state.sr(), sample_rate,
                                       resampling_method="sinc_interp_hann", lowpass_filter_width=16)

        return enhanced_tensor

    def _get_denoised_slice(self, denoised_samples: torch.Tensor, start_idx: int, end_idx: Optional[int]) -> bytes:
        enhanced_chunk: torch.Tensor = denoised_samples.squeeze()[start_idx:end_idx]
        output_buffer: torch.Tensor = (enhanced_chunk * (1 << 15)).to(torch.int16)
        return output_buffer.numpy().tobytes()
