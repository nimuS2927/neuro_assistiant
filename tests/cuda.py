import unittest
import torch


class TestCUDAAvailability(unittest.TestCase):
    def test_cuda_availability(self):
        # Проверка доступности CUDA
        self.assertTrue(torch.cuda.is_available(), "CUDA is not available")


if __name__ == "__main__":
    unittest.main()
