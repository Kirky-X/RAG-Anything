import unittest
from unittest.mock import patch

import raganything.models.device as dev
from raganything.models.device import DeviceManager


class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        # Reset singleton before each test
        DeviceManager._instance = None

    def test_singleton(self):
        """Test that DeviceManager follows singleton pattern."""
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        self.assertIs(dm1, dm2)
        # Note: device_manager global instance might have been created before reset
        # so we re-instantiate or check logic

    def test_device_properties(self):
        """Test basic device properties."""
        dm = DeviceManager()
        self.assertIn(dm.device, ("cpu", "cuda", "mps"))
        mem = dm.get_available_memory()
        self.assertIsInstance(mem, float)
        self.assertGreaterEqual(mem, 0)

    def test_cpu_fallback_when_torch_unavailable(self):
        """Test fallback to CPU when torch is not available."""
        with patch.object(dev, "TORCH_AVAILABLE", False):
            # Recreate singleton
            DeviceManager._instance = None
            dm = DeviceManager()
            self.assertEqual(dm.device, "cpu")
            self.assertIsNone(dm.torch_device)
            self.assertFalse(dm.is_gpu_available())

    def test_selects_cuda_when_available(self):
        """Test CUDA selection when available."""

        class FakeTensor:
            def cuda(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(_idx):
                return "FakeGPU"

            @staticmethod
            def memory_allocated(_idx):
                return 0

            @staticmethod
            def memory_reserved(_idx):
                return 0

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FakeTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "cuda")
                self.assertTrue(dm.is_gpu_available())

    def test_selects_mps_when_available(self):
        """Test MPS selection when available (macOS)."""

        class FakeTensor:
            def cuda(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return True

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FakeTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "mps")
                self.assertTrue(dm.is_gpu_available())

    def test_init_failure_fallback(self):
        """Test fallback to CPU if GPU init fails."""

        class FailingTensor:
            def cuda(self):
                raise RuntimeError("cuda fail")

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FailingTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "cpu")


if __name__ == "__main__":
    unittest.main()
