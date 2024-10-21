
import time

queue_task_id = ''

def set_global_queue_task_id(task_id):
    global queue_task_id
    queue_task_id = task_id

def get_global_queue_task_id():
    return queue_task_id

api_call = False

def set_global_api_call(api_call_):
    global api_call
    api_call = api_call_

def get_global_api_call():
    return api_call

vae = None

def set_global_vae(vae_):
    global vae
    vae = vae_

def get_global_vae():
    return vae

global_func = None

def store_function(func):
    global global_func
    global_func = func

def get_function():
    return global_func

def build_openapi_item(json_data, pull_task, api_call=False):
    openapi_item = {
        "callback_url": json_data["callback_url"] if "callback_url" in json_data else None,
        "origin_callback_url": json_data["origin_callback_url"] if "origin_callback_url" in json_data else None,
        "created": time.time(),
        "model": json_data["openapi_extra_data"]["model"] if "openapi_extra_data" in json_data and "model" in json_data["openapi_extra_data"] else None,
        "sync": json_data["sync"] if "sync" in json_data else None,
        "pull_task": pull_task,
        "api_call": set_global_api_call(api_call),
        "queue_task_id": get_global_queue_task_id()
    }
    return openapi_item

def queue_get_detail_request(queue_task_id):
    req = {"task_id": queue_task_id}
    return req

def queue_update_request(queue_task_id, status, data=""):
    req = {"task_id": queue_task_id, "status": status, "output_data": data}
    return req