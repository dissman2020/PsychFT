import os


def get_model_name(name_or_path):
    return os.path.basename(name_or_path.rstrip("/"))


def get_file_path(dir, name_or_path):
    import time
    current_time = time.strftime('_%Y-%m-%d_%H%M', time.localtime(time.time()))
    file_name = get_model_name(name_or_path) + current_time + '.csv'
    return os.path.join(dir, file_name)
