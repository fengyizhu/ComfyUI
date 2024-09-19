
pull_task = False

def set_pull_task(pull_task_: bool):
    global pull_task
    pull_task = pull_task_

def get_pull_task() -> bool:
    return pull_task

queue_task_id = ''

def set_queue_task_id(task_id):
    global queue_task_id
    queue_task_id = task_id

def get_queue_task_id():
    return queue_task_id

sync = False

def set_sync(sync_: bool):
    global sync
    sync = sync_

def get_sync() -> bool:
    return sync