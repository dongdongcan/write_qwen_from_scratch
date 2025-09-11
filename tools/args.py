import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", required=True)
    parser.add_argument("--verbose", "-V", action="store_true", help="show debug info", default=False)
    parser.add_argument("--max_new_tokens", type=int, help="max supported generation token numbers", default=30)
    args = parser.parse_args()
    return args
