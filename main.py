import signal
import sys
import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
import json
import requests

from logger import set_request_context
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys

from comfy_worker import ComfyWorker, WorkerConfig

if __name__ == "__main__":
    #NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

from openapi_utils import get_global_queue_task_id, queue_update_request, build_openapi_item, set_global_queue_task_id

# setup_logger(log_level=args.verbose)
# setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

def apply_custom_paths():
    # extra model paths
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    # --output-directory, --input-directory, --user-directory
    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")

apply_custom_paths()
execute_prestartup_script()


# Main code
import asyncio
import shutil
import threading
import gc


if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc

import comfy.utils

import execution
import server
from server import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0

def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")

model_list_str = os.getenv('MODEL_LIST', '')
model_list = model_list_str.split(',')
endpoint = os.getenv('ENDPOINT', "/v1/images/ke/generations")
TASK_FAILED = "failed"
TASK_COMPLETED = "completed"
TASK_IN_PROGRESS = "in_progress"
TASK_CANCELLED = "cancelled"

def post_request(url, body):
    response = requests.post(url=url, json=body)
    return response

def handle_failed_execution(e, item, pull_task, task_id):
    err = e.status_messages[2][1]
    response = dict(code=err['code'], message=err['exception_message'], node_id=err['node_id'],
                        timestamp=int(time.time()), task_id=task_id, openapi_item=item[6])
    queue_response = None
    if pull_task:
        queue_response = queue_update_request(get_global_queue_task_id(), TASK_FAILED, response)
    return response, queue_response

def handle_successful_execution(e, item, pull_task, task_id):
    response = None
    for key, value in e.history_result['outputs'].items():
        if "openapi_data" in value:
            output_data = value["openapi_data"][0]
            output_data["task_id"] = task_id
            output_data["openapi_item"] = item[6]
            response = output_data
    queue_response = None
    if pull_task:
        if output_data.get('pull_task_data'):
            update_status_data = json.dumps(output_data['pull_task_data'])
            queue_response = queue_update_request(get_global_queue_task_id(), TASK_COMPLETED, update_status_data)
        else:
            queue_response = None

    return response, queue_response


def handle_execution_result(e, item, server, update_status_url, task_id):
    if server.client_id is not None:
        prompt_id = item[1]
        server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)
    call_back = item[5]
    callback_url = item[6]['callback_url']
    sync = item[6]['sync']
    pull_task = item[6]['pull_task']

    if not e.success:
        resp, queue_resp = handle_failed_execution(e, item, pull_task, task_id)
    else:
        resp, queue_resp = handle_successful_execution(e, item, pull_task, task_id)

    # if pull_task:
    #     post_request(update_status_url, queue_resp)
    if sync:
        call_back.put(resp)
    if not sync and callback_url:
        response = post_request(callback_url, resp)
        logging.info(f"Send callback url: {callback_url} , Response status: {response.status_code}")


def process_queue_item(queue_item, q, e, server_instance, update_status_url):
    execution_start_time = time.perf_counter()
    if q.get_current_queue_length() > 0:
        logging.info(f"Queue pending length is {q.get_current_queue_length()}")

    item, item_id = queue_item
    set_request_context(item[3]['client_id'])
    task_id = item[3]['client_id']
    prompt_id = item[1]
    server_instance.last_prompt_id = prompt_id
    logging.info(f"Execute task and wait for {time.time() - item[6]['created']} seconds")
    
    server_instance.last_prompt_id = prompt_id
    e.execute(item[2], prompt_id, item[3], item[4])
    need_gc = True
    q.task_done(item_id,
                e.history_result,
                status=execution.PromptQueue.ExecutionStatus(
                    status_str='success' if e.success else 'error',
                    completed=e.success,
                    messages=e.status_messages))
    if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

    handle_execution_result(e, item, server_instance, update_status_url, task_id)

    current_time = time.perf_counter()
    execution_time = current_time - execution_start_time
    logging.info("Prompt executed in {:.2f} seconds".format(execution_time))
    return need_gc

def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=args.cache_lru)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0
    update_status_url = args.update_task_status_url if args.update_task_status_url else None
    time.sleep(15)

    while True:
        if not server_instance.task_loop:
            time.sleep(10)
            continue
        try:
            timeout = 0.5
            if need_gc:
                timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

            queue_item = q.get(timeout=timeout)
            if queue_item is not None:
                need_gc = process_queue_item(queue_item, q, e, server_instance, update_status_url)
            
            flags = q.get_flags()
            free_memory = flags.get("free_memory", False)

            if flags.get("unload_models", free_memory):
                comfy.model_management.unload_all_models()
                need_gc = True
                last_gc_collect = 0

            if free_memory:
                e.reset()
                need_gc = True
                last_gc_collect = 0

            # if args.free_memory and not free_memory:
            #     e.reset()

            if need_gc:
                start = time.time()
                current_time = time.perf_counter()
                if (current_time - last_gc_collect) > gc_collect_interval:
                    # comfy.model_management.cleanup_models()
                    gc.collect()
                    comfy.model_management.soft_empty_cache()
                    last_gc_collect = current_time
                    need_gc = False
                logging.info("GC took {:.2f} seconds".format(time.time() - start))

            if args.get_task:
                    get_task(q, server_instance)

            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()
        except Exception as err:
            logging.error("Error in prompt worker: {}".format(err))
            time.sleep(30)
            e.reset()
            gc.collect()
            current_time = time.perf_counter()
            continue

def get_task(q, server):
    if args.get_task_url: 
        get_url = args.get_task_url
    if args.get_task_detail_url:
        get_detail_url = args.get_task_detail_url
    if args.update_task_status_url:
        update_status_url = args.update_task_status_url
    # post queue
    req = {
        "endpoint": endpoint,
        "models": model_list,
        "size": 1,
        "level": 0
    }

    try:
        response = requests.post(url=get_url, json=req).json()
    except requests.ConnectTimeout as err:
        logging.error(f"Failed to get prompt from server: {err}")
        return

    if response['data'] == []:
        try:
            req['level'] = 1
            response = requests.post(url=get_url, json=req).json()
        except requests.ConnectTimeout as err:
            logging.error(f"Failed to get prompt from server: {err}")
            return
        if response['data'] == []:
            return

    response = response['data'][0]
    queue_task_id = response['task_id']
    set_global_queue_task_id(queue_task_id)
    status = response['status']
    input_data = response['input_data']
    input_file_id = response['input_file_id']
    ak = response['ak']

    if input_file_id is not None and input_file_id != "":
        # todo file
        req = queue_update_request(queue_task_id, TASK_IN_PROGRESS, "")
        resp = post_request(update_status_url, req)
        if resp.status_code != 200:
            logging.error(f"Failed to update status to in_progress: {resp.status_code}")
        return
        openai.api_key = ak
        response = openai.files.retrieve(input_file_id)


    json_data = input_data
    if "client_id" in json_data:
        task_id = json_data['client_id']
        set_request_context(json_data['client_id'])
        logging.info(f"got prompt, task id: {json_data['client_id']}")

    if "number" in json_data:
        number = float(json_data['number'])
    else:
        number = server.number
        if "front" in json_data:
            if json_data['front']:
                number = -number

        server.number += 1

    if "prompt" in json_data:
        prompt = json_data["prompt"]
        valid = execution.validate_prompt(prompt)
        extra_data = {}
        if "extra_data" in json_data:
            extra_data = json_data["extra_data"]

        if "client_id" in json_data:
            extra_data["client_id"] = json_data["client_id"]

        if valid[0]:
                # prompt_id = str(uuid.uuid4())
                prompt_id = task_id
                outputs_to_execute = valid[2]

    if "sync" in json_data:
        json_data["sync"] = False
    openapi_item = build_openapi_item(json_data, True, True)

    req = queue_update_request(queue_task_id, TASK_IN_PROGRESS, "")
    resp = post_request(update_status_url, req)
    if resp.status_code != 200:
        logging.error(f"Failed to update status to in_progress: {resp.status_code}")

    q.put((number, prompt_id, prompt, extra_data, outputs_to_execute, None, openapi_item))


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    def hook(value, total, preview_image):
        comfy.model_management.throw_exception_if_processing_interrupted()
        progress = {"value": value, "max": total, "prompt_id": server_instance.last_prompt_id, "node": server_instance.last_node_id}

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            server_instance.send_sync(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE, preview_image, server_instance.client_id)

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def signal_handler(sig, frame):
    from comfy.cli_args import args
    logging.info("Received signal: {}".format(sig))
    flags = args.get_flags()
    handle_signal = flags.get("handle_signal", False)
    logging.info("Handle signal: {}".format(handle_signal))
    if handle_signal:
        logging.info("Received signal, system exit .")
        sys.exit(0)


def setup_database():
    try:
        from app.database.db import init_db, dependencies_available
        if dependencies_available():
            init_db()
    except Exception as e:
        logging.error(f"Failed to initialize database. Please ensure you have installed the latest requirements. If the error persists, please report this as in future the database will be required: {e}")


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)
    prompt_server = server.PromptServer(asyncio_loop)

    hook_breaker_ac10a0.save_functions()
    nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    )
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    setup_database()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    def monitor_thread(q, server):
        while True:
            worker_thread = threading.Thread(target=prompt_worker, args=(q, server,), daemon=True)
            worker_thread.start()
            worker_thread.join()

    threading.Thread(target=monitor_thread, args=(prompt_server.prompt_queue, prompt_server,), daemon=True).start()

    # 启动ComfyWorker
    comfy_worker = ComfyWorker(
        WorkerConfig.from_env(),
        prompt_server,
        endpoint=endpoint,
        cache_lru=args.cache_lru,
        cache_none=args.cache_none
    )
    comfy_worker.start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))
    signal.signal(signal.SIGTERM, signal_handler)

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")

    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
