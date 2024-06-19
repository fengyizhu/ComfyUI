import gc
from collections import OrderedDict
import os
import logging
import sys
import time

import psutil
import torch

from comfy.cli_args import args

current_device = torch.cuda.current_device()
gpu_total_memory = torch.cuda.get_device_properties(current_device).total_memory
cpu_total_memeory = psutil.virtual_memory().total

class ModelCache:
    def __init__(self, cpu_device_size_ratio, gpu_device_size_ratio, cache_state=True):
        self.cpu_cache = OrderedDict()
        self.cpu_cache_time = OrderedDict()
        self.cpu_device_size = cpu_total_memeory / (1024 * 1024) * cpu_device_size_ratio
        self.gpu_device_size = gpu_total_memory * gpu_device_size_ratio
        self.gpu_cache = []
        self.cpu_cache_size = 0
        self._cache_state = cache_state


    def _get_cache_model(self, key):
        if self.cpu_cache.get(key) is None:
            self.cpu_cache[key] = {}

        return self.cpu_cache[key]


    def _get_cache_model_time(self, key):
        if self.cpu_cache_time.get(key) is None:
            self.cpu_cache_time[key] = time.time()

        return self.cpu_cache_time[key]


    def is_model_in_cache(self, key):
        return key in self.cpu_cache


    def current_cpu_device_size(self):
        return psutil.virtual_memory().used


    def current_gpu_device_size_ratio(self):
        return torch.cuda.memory_allocated(current_device) / gpu_total_memory


    def _check_model_size(self, model):
        if sys.getsizeof(model) >= self.cpu_device_size:
            logging.error("Model too large for device")
            raise Exception("Model too large for device")
        
        while self.cpu_cache_size + sys.getsizeof(model) >= self.cpu_device_size:
            logging.info("Cache full, removing least recently used model")
            self.unload_last_model()


    def _cache_model(self, key, model, prop):
        if not self._cache_state:
            return
        
        self._check_model_size(model)

        self.cpu_cache_size += sys.getsizeof(model)
        logging.info(f"Adding {prop} model to cache")
        self._get_cache_model(key)[prop] = model
        self._get_cache_model_time(key)


    def cache_model(self, key, model):
        self._cache_model(key, model, 'model')


    def cache_clip_vision(self, key, clip_vision):
        self._cache_model(key, clip_vision, 'clip_vision')


    def cache_vae(self, key, vae):
        self._cache_model(key, vae, 'vae')


    def cache_sd(self, key, sd):
        assert isinstance(sd, dict)
        keys = list(sd.keys())
        values = list(sd.values())
        self._cache_model(key, (keys, values), 'sd')


    def cache_clip(self, key, clip_key, clip):
        self._cache_model(key, clip, clip_key)


    def refresh_cache(self, key):
        if key in self.cpu_cache:
            self.cpu_cache.move_to_end(key)

    def get_model(self, key, prop):
        item = self.cpu_cache.get(key)
        if item is None:
            logging.info(f"Model not in cache {key}")
            return None

        logging.info(f"Retrieved model from cache {key}")
        self.cpu_cache_time[key] = time.time()

        if prop == "sd":
            if item.get('sd') is None:
                return None
            k, values = item.get("sd")
            return dict(zip(k, values))
        return item.get(prop)
    
    def __size__(self):
        return sys.getsizeof(self.cpu_cache)

    def __len__(self):
        return len(self.cpu_cache)

    @property
    def cache(self):
        return self.cpu_cache

    @staticmethod
    def unpatch_offload_model(model):
        model.model_patches_to(model.offload_device)


    def unload_last_model(self):
        if self.cpu_cache_size == 0:
            return

        unload_key = min(self.cpu_cache_time, key=self.cpu_cache_time.get)
        cache_k, item = self.cpu_cache.pop(unload_key)
        self.cpu_cache_time.pop(unload_key)
        self.cpu_cache_size = self.__size__()
        item.pop("sd", None)

        for k in list(item.keys()):
            model = item.pop(k, None)
            if model is not None:
                if hasattr(model, "patcher"):
                    self.unpatch_offload_model(model.patcher)
                else:
                    self.unpatch_offload_model(model)

        del item
        model_dir, model_name = os.path.split(cache_k)
        dir_name = os.path.basename(model_dir)
        gc.collect()
        logging.info(f"Drop model cache: {model_name} ({dir_name})")


model_cache = ModelCache(0.8,0.8)