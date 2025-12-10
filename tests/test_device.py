import unittest
from raganything.models.device import DeviceManager, device_manager

class TestDeviceManager(unittest.TestCase):
    def test_singleton(self):
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        self.assertIs(dm1, dm2)
        self.assertIs(dm1, device_manager)

    def test_device_properties(self):
        dm = device_manager
        self.assertIn(dm.device, ("cpu", "cuda", "mps"))
        mem = dm.get_available_memory()
        self.assertIsInstance(mem, float)
        self.assertGreater(mem, 0)

if __name__ == "__main__":
    unittest.main()
