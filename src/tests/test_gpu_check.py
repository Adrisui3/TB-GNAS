import torch
import pytest

class TestTorchGPU:
    def test_is_cuda_enabled(self):
        assert torch.cuda.is_available()

    def test_gpus_available(self):
        assert torch.cuda.device_count() >= 1