import unittest
from unittest.mock import MagicMock, patch
import torch
import psutil
from collections import OrderedDict
from pympler import asizeof
from model_cache import ModelCache

class TestModelCache(unittest.TestCase):

    def setUp(self):
        self.cpu_device_size_ratio = 0.1
        self.gpu_device_size_ratio = 0.1
        self.process_counts = 1
        self.gpu_limit_ratio = 0.5
        self.model_cache = ModelCache(self.cpu_device_size_ratio, self.gpu_device_size_ratio, self.process_counts, self.gpu_limit_ratio)

    def test_initialization(self):
        self.assertIsInstance(self.model_cache.cpu_cache, OrderedDict)
        self.assertIsInstance(self.model_cache.cpu_cache_time, OrderedDict)
        self.assertIsInstance(self.model_cache.gpu_cache, list)
        self.assertIsInstance(self.model_cache.gpu_cache_time, OrderedDict)
        self.assertEqual(self.model_cache.gpu_limit_ratio, self.gpu_limit_ratio)

    @patch('torch.cuda.memory_allocated', return_value=1024 * 1024 * 100)  # 100 MB
    @patch('torch.cuda.get_device_properties')
    def test_current_gpu_device_size_ratio(self, mock_get_device_properties, mock_memory_allocated):
        mock_get_device_properties.return_value.total_memory = 1024 * 1024 * 1000  # 1 GB
        ratio = self.model_cache.current_gpu_device_size_ratio()
        self.assertNotEqual(ratio, 1)

    def test_cache_model(self):
        model = MagicMock()
        self.model_cache.cache_model('test_model', model)
        self.assertIn('test_model', self.model_cache.cpu_cache)
        self.assertEqual(self.model_cache.cpu_cache['test_model']['model'], model)

    def test_get_cpu_model(self):
        model = MagicMock()
        self.model_cache.cache_model('test_model', model)
        cached_model = self.model_cache.get_cpu_model('test_model', 'model')
        self.assertEqual(cached_model, model)

    def test_unload_last_cpu_model(self):
        model = MagicMock()
        self.model_cache.cache_model('test_model_1', model)
        self.model_cache.cache_model('test_model_2', model)
        self.model_cache.unload_last_cpu_model()
        self.assertNotIn('test_model_1', self.model_cache.cpu_cache)
        self.assertIn('test_model_2', self.model_cache.cpu_cache)

if __name__ == '__main__':
    unittest.main()