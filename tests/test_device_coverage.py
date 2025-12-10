import unittest
from unittest.mock import patch


class TestDeviceManagerCoverage(unittest.TestCase):
    def test_cpu_fallback_when_torch_unavailable(self):
        import raganything.models.device as dev

        with patch.object(dev, "TORCH_AVAILABLE", False):
            # Recreate singleton to trigger detection with patched flag
            dev.DeviceManager._instance = None
            dm = dev.DeviceManager()
            self.assertEqual(dm.device, "cpu")
            self.assertIsNone(dm.torch_device)
            self.assertFalse(dm.is_gpu_available())

    def test_selects_cuda_when_available(self):
        import raganything.models.device as dev

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
                dev.DeviceManager._instance = None
                dm = dev.DeviceManager()
                self.assertEqual(dm.device, "cuda")
                self.assertTrue(dm.is_gpu_available())
                self.assertIsNotNone(dm.torch_device)

    def test_selects_mps_when_available(self):
        import raganything.models.device as dev

        class FakeTensor:
            def cuda(self):
                return self
            def to(self, *_args, **_kwargs):
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return False
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
                dev.DeviceManager._instance = None
                dm = dev.DeviceManager()
                self.assertEqual(dm.device, "mps")
                self.assertTrue(dm.is_gpu_available())
                self.assertIsNotNone(dm.torch_device)

    def test_cuda_available_but_init_failure(self):
        import raganything.models.device as dev

        class FailingTensor:
            def cuda(self):
                raise RuntimeError("cuda init fail")

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
                dev.DeviceManager._instance = None
                dm = dev.DeviceManager()
                self.assertEqual(dm.device, "cpu")

    def test_mps_available_but_init_failure(self):
        import raganything.models.device as dev

        class FailingTensor:
            def to(self, *_args, **_kwargs):
                raise RuntimeError("mps init fail")

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
                return FailingTensor()
            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                dev.DeviceManager._instance = None
                dm = dev.DeviceManager()
                self.assertEqual(dm.device, "cpu")


if __name__ == "__main__":
    unittest.main()
