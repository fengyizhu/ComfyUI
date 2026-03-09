"""ComfyUI Worker - 使用Bella Queue SDK的TaskProcessor模式"""
import asyncio
import logging

import requests
from bella_openapi import BellaWorker, WorkerConfig
from bella_openapi.worker.models import QueueTask

from logger import set_request_context

import execution
from openapi_utils import set_global_queue_task_id, build_openapi_item, get_global_pull_task_tag, set_global_pull_task_tag


class ComfyWorker:

    def __init__(self, config: WorkerConfig, server_instance, endpoint: str, cache_lru: int = 0, cache_none: bool = False, queue: execution.PromptQueue = None):
        """
        初始化ComfyWorker
        """
        self.queue = queue
        self.config = config
        self.server_instance = server_instance
        self.endpoint = endpoint
        self.logger = logging.getLogger(__name__)

        # 根据参数配置cache类型和大小
        if cache_none:
            cache_type = execution.CacheType.DEPENDENCY_AWARE
            cache_size = None
        elif cache_lru > 0:
            cache_type = execution.CacheType.LRU
            cache_size = cache_lru
        else:
            # 默认使用 CLASSIC，传递 False（匹配默认值）
            cache_type = False
            cache_size = None

        # 创建PromptExecutor实例
        self.executor = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=cache_size)

        # 创建BellaWorker实例（组合而非继承）
        self.worker = BellaWorker(
            host=config.host,
            api_key=config.api_key,
            queues=config.queues,
            poll_interval=config.poll_interval,
            batch_size=config.batch_size,
            max_concurrent_tasks=config.max_concurrent_tasks,
            strategy=config.strategy
        )

        # 注册任务处理器（注册包装方法）
        self.worker.register_task_processor(
            endpoint=self.endpoint,
            processor=self._process_task_with_callback
        )

        self.logger.info(f"ComfyWorker initialized with endpoint: {self.endpoint}")

    def start(self):
        """启动Worker（委托给BellaWorker）"""
        self.logger.info("Starting ComfyWorker...")
        self.worker.start()

    def stop(self):
        """停止Worker（委托给BellaWorker）"""
        self.logger.info("Stopping ComfyWorker...")
        self.worker.stop()

    async def _process_task_with_callback(self, task: QueueTask) -> dict[str, object]:
        """任务处理包装器：执行任务并发送回调"""

        json_data = task.data
        if "client_id" in json_data:
            task_id = json_data['client_id']
            set_request_context(json_data['client_id'])
            logging.info(f"got prompt, task id: {json_data['client_id']}")

        if "number" in json_data:
            number = float(json_data['number'])
        else:
            number = self.server_instance.number
            if "front" in json_data:
                if json_data['front']:
                    number = -number

            self.server_instance.number += 1

        if "prompt" in json_data:
            prompt = json_data["prompt"]
            prompt_id = task_id if task_id else str(uuid.uuid4())
            partial_execution_targets = None
            if "partial_execution_targets" in json_data:
                partial_execution_targets = json_data["partial_execution_targets"]
            valid = await execution.validate_prompt(prompt_id, prompt, partial_execution_targets)
            extra_data = {}
            if "extra_data" in json_data:
                extra_data = json_data["extra_data"]

            if "client_id" in json_data:
                extra_data["client_id"] = json_data["client_id"]

            if valid[0]:
                    # prompt_id = str(uuid.uuid4())
                    # prompt_id = task_id
                    outputs_to_execute = valid[2]

        if "sync" in json_data:
            json_data["sync"] = False
        openapi_item = build_openapi_item(json_data, True, True)

        set_global_pull_task_tag(True)
        self.queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute, None, openapi_item))
        while get_global_pull_task_tag():
            await asyncio.sleep(0.1)

        result = None
        return result