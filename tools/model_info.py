from transformers.utils import cached_file
import os


def get_mode_cached_dir(model_name):
    try:
        json_file = cached_file(model_name, "config.json")
        return os.path.dirname(json_file)
    except:
        print(f"Model {model_name} not found in local cache.")
        return None
