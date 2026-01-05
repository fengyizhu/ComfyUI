"""ComfyUI Worker - 使用Bella Queue SDK的TaskProcessor模式"""
import asyncio
import logging

import requests
from bella_openapi import BellaWorker, WorkerConfig
from bella_openapi.worker.models import QueueTask

import execution
from openapi_utils import set_global_queue_task_id


class ComfyWorker:

    def __init__(self, config: WorkerConfig, server_instance, endpoint: str, cache_lru: int = 0, cache_none: bool = False):
        """
        初始化ComfyWorker
        """
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
        result = await self._process_task(task)

        # 发送回调（异步非阻塞，不等待回调完成）
        callback_url = task.data.get("callback_url")
        if callback_url:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None,
                lambda: requests.post(callback_url, json={"task_id": task.task_id, "result": result}, timeout=30)
            )
        return result

    async def _process_task(self, task: QueueTask) -> dict[str, object]:
        """处理单个ComfyUI任务（纯任务处理逻辑）"""
        queue_task_id = task.task_id
        data = task.data

        # 设置全局任务ID上下文
        set_global_queue_task_id(queue_task_id)

        try:
            # 1. 验证任务数据
            if "prompt" not in data:
                self.logger.error(f"No prompt in task {queue_task_id}")
                result = {"error": "No prompt in task data"}
                return result

            # 2. 验证prompt结构
            valid = execution.validate_prompt(data["prompt"])
            if not valid[0]:
                self.logger.error(f"Invalid prompt for task {queue_task_id}: {valid[1]}")
                result = {"error": valid[1], "node_errors": valid[3]}
                return result

            # 3. 准备任务数据
            task_id = data.get('client_id', queue_task_id)
            prompt = data["prompt"]

            # 准备extra_data
            extra_data = data.get("extra_data", {})
            if "client_id" in data:
                extra_data["client_id"] = data["client_id"]

            # 输出节点
            outputs_to_execute = valid[2]

            self.logger.info(f"Executing task {task_id} (queue_task_id: {queue_task_id})")

            # 4. 直接执行工作流（同步等待完成）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.executor.execute,
                prompt,
                task_id,
                extra_data,
                outputs_to_execute
            )

            # 5. 检查执行结果
            if not self.executor.success:
                error_msg = "\n".join(self.executor.status_messages) if self.executor.status_messages else "Execution failed"
                self.logger.error(f"Task {task_id} failed: {error_msg}")
                result = {
                    "error": error_msg,
                    "node_errors": self.executor.history_result.get("node_errors", {})
                }
                return result

            # 6. 返回成功结果
            self.logger.info(f"Task {task_id} completed successfully")
            result = self.executor.history_result
            return result

        except asyncio.CancelledError:
            self.logger.warning(f"Task {queue_task_id} cancelled")
            return {"error": "Task cancelled"}
        except Exception as e:
            self.logger.error(f"Error processing task {queue_task_id}: {e}", exc_info=True)
            return {"error": str(e)}
