#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from typing import Callable, Dict, List, Union


def is_python(filename: str) -> bool:
    return filename.lower().endswith(".py")


def find_all_files_under_dir(abs_path: str) -> List[str]:
    """Recursively get all files under abs_path."""
    file_list = []
    try:
        for root, _, files in os.walk(abs_path):
            for file in files:
                full_path = os.path.join(root, file)
                file_list.append(full_path)
        return file_list
    except Exception as e:
        print(f"query files in {abs_path} fail: {e}")
        return []


def find_all_git_diff_files_under_dir() -> List[str]:
    """Recursively get all files under abs_path."""
    cmd = ["git", "diff", "--name-only"]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    modified_files = result.stdout.splitlines()
    return [f for f in modified_files if os.path.isfile(f)]


def find_source_files(args, filter_func: Callable[[str], bool]) -> List[str]:
    source_file = [args.file] if args.file else find_all_git_diff_files_under_dir()
    return [f for f in source_file if filter_func(f)]


def process_files(args, filter_to_commands: Dict[Callable[[str], bool], Union[Callable[[str], bool], List[str]]]):
    def run_command(command: List[str]) -> bool:
        try:
            print(f"{' '.join(command)}")
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"cmd {command} failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"tool {command[0]} not usable")
            return False

    exit_code = 0
    for filter_func, cmd in filter_to_commands.items():
        files = find_source_files(args, filter_func)
        if not files:
            continue

        for file in files:
            if isinstance(cmd, Callable):
                if not cmd(file):
                    print(f"deal {file} fail")
                    exit_code = 1
            else:
                if not run_command(cmd + [file]):
                    exit_code = 1

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="format code")
    parser.add_argument("--file", type=str, help="A single file to be formatted")
    args = parser.parse_args()

    filter_to_commands = {
        is_python: ["black", "-q", "--line-length", "120"],
    }

    process_files(args, filter_to_commands)
