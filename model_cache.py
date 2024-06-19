import gc
from collections import OrderedDict
import os
import logging
import sys

import psutil
import GPUtil

from comfy.cli_args import args


class ModelCache:
    def __init__(self):
        self._cache = OrderedDict()
        # ignore gpu and highvram state, may cause OOM error
        self.cpu_device_size = psutil.virtual_memory().total * 0.8
        self.gpu_device_size = GPUtil.getGPUs()[3].memoryTotal
        self.gpu_cache = []
        self.cache_size = 0
        self._cache_state = True

    def _get_item(self, key):
        if self._cache.get(key) is None:
            self._cache[key] = {}

        return self._cache[key]

    def is_model_in_cache(self, key):
        return key in self.cache_pool
    
    def unload_last_model(self):
        if self.cache_size == 0:
            return

        self.cache_size -= sys.getsizeof(self.cache_pool.peekitem(last=False)[1])
        self.free_one_model_cache()

    def current_device_size(self):
        return psutil.virtual_memory().total - self.cache_size

    def current_gpu_device_size(self):
        return GPUtil.getGPUs()[3].memoryTotal.memoryUsed
    
    def check_model_size(self, model):
        if sys.getsizeof(model) >= self.cpu_device_size:
            logging.error("Model too large for device")
            raise Exception("Model too large for device")
        
        while self.cache_size + sys.getsizeof(model) >= self.cpu_device_size:
            logging.info("Cache full, removing least recently used model")
            self.unload_last_model()

    def cache_model(self, key, model):
        if not self._cache_state:
            return
        
        self.check_model_size(model)
        
        self.cache_size += sys.getsizeof(model)
        logging.info(f"Adding model to cache")
        item = self._get_item(key)
        item['model'] = model

    def cache_clip_vision(self, key, clip_vision):
        if not self._cache_state:
            return
        
        self.check_model_size(clip_vision)

        self.cache_size += sys.getsizeof(clip_vision)
        logging.info(f"Adding model to cache")
        item = self._get_item(key)
        item['clip_vision'] = clip_vision

    def cache_vae(self, key, vae):
        if not self._cache_state:
            return
        
        self.check_model_size(vae)

        self.cache_size += sys.getsizeof(vae)
        logging.info(f"Adding model to cache {key}")
        item = self._get_item(key)
        item['vae'] = vae

    def cache_sd(self, key, sd):
        if not self._cache_state:
            return
        assert isinstance(sd, dict)
        keys = list(sd.keys())
        values = list(sd.values())

        logging.info(f"Adding model to cache {key}")
        item = self._get_item(key)
        item['sd'] = (keys, values)

    def cache_clip(self, key, clip_key, clip):
        self.check_model_size(clip)

        self.cache_size += sys.getsizeof(clip)
        logging.info(f"Adding model to cache {key}")
        item = self._get_item(key)
        item[clip_key] = clip

    def refresh_cache(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)

    def get_item(self, key, prop):
        item = self._cache.get(key)
        if item is None:
            logging.info(f"Model not in cache {key}")
            return None

        self.refresh_cache(key)
        logging.info(f"Retrieved model from cache {key}")

        if prop == "sd":
            if item.get('sd') is None:
                return None
            k, values = item.get("sd")
            return dict(zip(k, values))
        return item.get(prop)
    
    def __size__(self):
        return sys.getsizeof(self._cache)

    def __len__(self):
        return len(self._cache)

    @property
    def cache(self):
        return self._cache

    @staticmethod
    def unpatch_offload_model(model):
        model.model_patches_to(model.offload_device)

    def unload_last_model(self):
        if self.cache_size == 0:
            return

        self.cache_size -= sys.getsizeof(self.cache_pool.peekitem(last=False)[1])
        cache_k, item = self._cache.popitem(last=False)
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


model_cache = ModelCache()