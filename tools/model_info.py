from transformers.utils import cached_file
import os


def get_mode_cached_dir(model_name):
    json_file = cached_file(model_name, "config.json")
    return os.path.dirname(json_file)
