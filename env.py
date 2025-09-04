#!/usr/bin/env python3
import argparse
import subprocess
import os
from typing import List, Literal, Sequence, Union
import shutil


def append_line_to_file(file_path: str, line: str) -> None:
    # Open the file in append mode ('a+')
    with open(file_path, "a+", encoding="utf-8") as file:
        file.write(line)
    print(f"Appended `{line}` to `{file_path}`")


def run_shell_cmd(cmd: Sequence[str], cwd: str, capture_output=False, silence=False):
    """Run specified command, and exit if error occurs"""
    if not silence:
        print(f"Running command `{' '.join(cmd)}` in {cwd}")

    result = subprocess.run(cmd, cwd=cwd, capture_output=capture_output, text=capture_output)
    if result.returncode:
        if capture_output:
            print(result.stdout)
            print(result.stderr)
        raise RuntimeError(f"Command `{' '.join(cmd)}` failed with error code {result.returncode}")
    return result.stdout.strip() if capture_output else None


def config_pip(args):
    upgrade_cmd = ["python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    run_shell_cmd(upgrade_cmd, args.root_path)

    config_cmd = ["pip", "config", "set", "global.index-url", "https://pypi.tuna.tsinghua.edu.cn/simple"]
    run_shell_cmd(config_cmd, args.root_path)


def update_package(args):
    install_hugging_face_cmd = ["pip3", "install", "-U", "huggingface_hub"]
    run_shell_cmd(install_hugging_face_cmd, args.root_path)

    install_hugging_face_cmd = ["pip", "install", "-U", "huggingface_hub[cli]"]
    run_shell_cmd(install_hugging_face_cmd, args.root_path)

    # install wheel package first, to make sure install package from requirements.txt successfully.
    # install_wheel_cmd = ["pip3", "install", "setuptools", "wheel"]
    # run_shell_cmd(install_wheel_cmd, args.root_path)

    install_package_cmd = ["pip3", "install", "-r", "requirements.txt"]
    run_shell_cmd(install_package_cmd, args.root_path)


def install_hooks(args):
    src_dir = os.path.join(args.root_path, "scripts")
    install_hook_cmd = ["git", "rev-parse", "--git-path", "hooks"]
    dst_dir = run_shell_cmd(install_hook_cmd, cwd=args.root_path, capture_output=True, silence=True)

    for file in os.listdir(src_dir):
        src_file = os.path.join(src_dir, file)
        if os.path.isfile(src_file) and file.startswith("pre-"):
            dst_file = os.path.join(args.root_path, dst_dir, file)
            shutil.copy(src_file, dst_file)
            os.chmod(dst_file, 0o755)
            print(file, "hook installed to", dst_file)


def create_venv_and_enter(args):
    env_path = os.path.join(args.root_path, args.env_name)
    if not os.path.exists(env_path) or args.rerun:
        cmd = ["python3", "-m", "venv", args.env_name]
        run_shell_cmd(cmd, args.root_path)
        # add some environment to .venv/bin/activate
        activate_file = os.path.join(env_path, "bin", "activate")
        hugging_face_end_point = "export HF_ENDPOINT=https://hf-mirror.com"
        append_line_to_file(activate_file, hugging_face_end_point)
        python_path = "export PYTHONPATH=$PYTHONPATH:" + args.root_path
        append_line_to_file(activate_file, python_path)
    else:
        print(f"{env_path} exist!")

    activate_path = os.path.join(args.root_path, args.env_name, "bin", "activate")
    print(f"Use the following cmd to enter virtual environment")
    print(f"\033[32m>>> source {activate_path}\033[0m")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="install all needed package")
    parser.add_argument("--rerun", action="store_true", help="force rerun setting env infos")

    args = parser.parse_args()
    args.env_name = ".venv"
    return args


def main():
    args = parse_args()
    args.script_path = os.path.abspath(__file__)
    assert os.path.exists(args.script_path)

    args.root_path = os.path.dirname(args.script_path)
    assert os.path.exists(args.root_path)

    install_hooks(args)
    create_venv_and_enter(args)

    if args.install:
        config_pip(args)
        update_package(args)


if __name__ == "__main__":
    main()
