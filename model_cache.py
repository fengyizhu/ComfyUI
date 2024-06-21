import gc
from collections import OrderedDict
from multiprocessing import process
import os
import logging
import sys
import time

import psutil
import torch

from pympler import asizeof

from comfy.cli_args import args

current_device = torch.cuda.current_device()
gpu_total_memory = torch.cuda.get_device_properties(current_device).total_memory
cpu_total_memeory = psutil.virtual_memory().total

class ModelCache:
    def __init__(self, cpu_device_size_ratio, gpu_device_size_ratio, process_counts=1, gpu_limit_ratio=0.5):
        self.cpu_cache = OrderedDict()
        self.cpu_cache_time = OrderedDict()
        self.cpu_device_size = cpu_total_memeory * cpu_device_size_ratio / process_counts
        self.gpu_device_size = gpu_total_memory * gpu_device_size_ratio
        self.gpu_cache = []
        self.gpu_cache_time = OrderedDict()
        self.gpu_limit_ratio = gpu_limit_ratio


    def __gpu_cache_size__(self):
        return asizeof.asizeof(self.gpu_cache)


    def __gpu_cache_len__(self):
        return len(self.gpu_cache)


    def current_gpu_device_size(self):
        return torch.cuda.memory_allocated(current_device)
    

    def current_gpu_device_size_ratio(self):
        return torch.cuda.memory_allocated(current_device) / gpu_total_memory


    def current_gpu_device_size_ratio_is_over(self):
        return self.current_gpu_device_size_ratio() > self.gpu_limit_ratio


    def put_gpu_cache(self, model, pos=0):
        current_gpu_device_size = self.current_gpu_device_size()

        while current_gpu_device_size > self.gpu_device_size:
            if self.gpu_cache == []:
                logging.warning(f"Current used gpu device size {current_gpu_device_size} over gpu device size {self.gpu_device_size}, and gpu cache is none, cannot cache model")
                return
            
            self.unload_last_gpu_model()
            logging.debug(f"Current used gpu device size {current_gpu_device_size} over gpu device size {self.gpu_device_size}, unload last model from cache")

        self.gpu_cache.insert(pos, model)
        self.gpu_cache_time[model.model] = time.time()


    def get_gpu_cache_model(self, model):
        cache_model = None

        try:
            cache_model = self.gpu_cache[model]
            logging.info(f"Retrieved model from gpu cache {model}")
        except:
            logging.info(f"Model not in cache {model}")
            cache_model = None

        self.gpu_cache_time[model.model] = time.time()
        return cache_model


    def unload_last_gpu_model(self):
        if self.__gpu_cache_len__() == 0:
            return

        unload_model = min(self.gpu_cache_time, key=self.gpu_cache_time.get)
        logging.debug(f"Unload model {unload_model} from cache")
        self.gpu_cache_time.pop(unload_model)

        for cached_model in self.gpu_cache:
            if cached_model.model == unload_model:
                index = self.gpu_cache.index(cached_model)
                need_unload_model = self.gpu_cache.pop(index)

                need_unload_model.model_unload(unpatch_weights=True)
                del need_unload_model
                break


    def __cpu_cache_size__(self):
        return asizeof.asizeof(self.cpu_cache)

    def __cpu_cache_len__(self):
        return len(self.cpu_cache)

    @property
    def cache(self):
        return self.cpu_cache

    @staticmethod
    def unpatch_offload_model(model):
        model.model_patches_to(model.offload_device)


    def is_model_in_cpu_cache(self, key):
        return key in self.cpu_cache


    def current_cpu_device_size(self):
        process = psutil.Process()
        return process.memory_info().rss


    def cache_model(self, key, model):
        self._put_cpu_cache_model(key, model, 'model')


    def cache_clip_vision(self, key, clip_vision):
        self._put_cpu_cache_model(key, clip_vision, 'clip_vision')


    def cache_vae(self, key, vae):
        self._put_cpu_cache_model(key, vae, 'vae')


    def cache_sd(self, key, sd):
        assert isinstance(sd, dict)
        keys = list(sd.keys())
        values = list(sd.values())
        self._put_cpu_cache_model(key, (keys, values), 'sd')


    def cache_clip(self, key, clip_key, clip):
        self._put_cpu_cache_model(key, clip, clip_key)


    def refresh_cache(self, key):
        if key in self.cpu_cache:
            self.cpu_cache.move_to_end(key)


    def _put_cpu_cache_model(self, key, model, prop):
        current_cpu_device_size = self.current_cpu_device_size()

        while current_cpu_device_size > self.cpu_device_size:
            if self.cpu_cache == {}:
                logging.warning(f"Current used cpu device size {current_cpu_device_size} over cpu device size {self.cpu_device_size}, and cpu cache is none, cannot cache model")
                return
            
            self.unload_last_cpu_model()
            logging.debug(f"Current used cpu device size {current_cpu_device_size} over cpu device size {self.cpu_device_size}, unload last model from cache")

        logging.info(f"Adding {prop}: {key} model to cache")
        self.cpu_cache.setdefault(key, {})[prop] = model
        self.cpu_cache_time[key] = time.time()


    def get_cpu_model(self, key, prop):
        item = self.cpu_cache.get(key)
        if item is None:
            logging.info(f"Model not in cache {prop} {key}")
            return None

        logging.info(f"Retrieved model from cpu cache {prop} {key} {self.cpu_cache_time.get(key)}")
        self.cpu_cache_time[key] = time.time()

        if prop == "sd":
            if item.get('sd') is None:
                return None
            k, values = item.get("sd")
            return dict(zip(k, values))
        return item.get(prop)


    def unload_last_cpu_model(self):
        if self.__len__() == 0:
            return

        unload_key = min(self.cpu_cache_time, key=self.cpu_cache_time.get)
        logging.debug(f"Unload model {unload_key} from cache")
        item = self.cpu_cache.pop(unload_key)
        self.cpu_cache_time.pop(unload_key)
        
        item.pop("sd", None)

        for k in list(item.keys()):
            model = item.pop(k, None)
            if model is not None:
                if hasattr(model, "patcher"):
                    self.unpatch_offload_model(model.patcher)
                else:
                    self.unpatch_offload_model(model)

        del item
        model_dir, model_name = os.path.split(unload_key)
        dir_name = os.path.basename(model_dir)
        gc.collect()
        logging.info(f"Drop model cache: {model_name} ({dir_name})")


model_cache = ModelCache(args.cpu_device_size_ratio, args.gpu_device_size_ratio, args.gpu_limit_ratio)
logging.info(f"Cpu device size ratio: {args.cpu_device_size_ratio}, Gpu device size ratio: {args.gpu_device_size_ratio}, Gpu limit ratio: {args.gpu_limit_ratio}")
logging.info(f"Cpu device size: {model_cache.cpu_device_size}, Gpu device size: {model_cache.gpu_device_size}")