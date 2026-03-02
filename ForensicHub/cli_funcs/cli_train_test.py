
import os
import sys
import re
import yaml
import subprocess
from colorama import init, Fore, Style
from IMDLBenCo.utils.paths import BencoPath
from .copy import copy_files, copy_file

from ForensicHub.training_scripts import train as train_script
from ForensicHub.training_scripts import test as test_script


def get_unique_log_dir(base_dir):
    """
    Generate a unique log directory path by appending _1, _2, etc. if the directory exists.

    Args:
        base_dir: The original log directory path from config

    Returns:
        A unique directory path that doesn't exist yet
    """
    if not os.path.exists(base_dir):
        return base_dir

    # Directory exists, find the next available suffix
    # First, check if base_dir already has a suffix like _1, _2, etc.
    match = re.match(r'^(.+)_(\d+)$', base_dir)
    if match:
        # Already has a suffix, extract base and start from next number
        base_path = match.group(1)
        start_num = int(match.group(2)) + 1
    else:
        # No suffix yet, start from _1
        base_path = base_dir
        start_num = 1

    # Find the next available number
    counter = start_num
    while True:
        new_dir = f"{base_path}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1


def check_config_dict(config):
    """
    TODO, 主要是校验字段是否合理，这里只进行简单校验，后续要完善
    """
    # check for 'gpus' 'log_dir' and 'flag' in config
    if 'gpus' not in config:
        print(Fore.RED + "  The config file must contain 'gpus'." + Style.RESET_ALL)
        return False
    if 'log_dir' not in config:
        print(Fore.RED + "  The config file must contain 'log_dir'." + Style.RESET_ALL)
        return False
    if 'flag' not in config:
        print(Fore.RED + "  The config file must contain 'flag'." + Style.RESET_ALL)
        return False
    return True

def excute_script(yaml_path, script_path):
    # Read the YAML file
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(Fore.RED + f"  Error reading YAML file: {e}" + Style.RESET_ALL)
        return
    except FileNotFoundError as e:
        print(Fore.RED + f"  The YAML file {yaml_path} does not exist." + Style.RESET_ALL)
        return
    except Exception as e:
        print(Fore.RED + f"  An unexpected error occurred: {e}" + Style.RESET_ALL)
        return
    
    # Check if the script is validate in our framework
    if not check_config_dict(config):
        print(Fore.RED + "  The config file is not valid." + Style.RESET_ALL)
        return
    
    # count gpu_count from gpus, it looks like "4,5" or "3"
    gpu_count = 0
    if isinstance(config['gpus'], str):
        gpu_count = len(config['gpus'].split(','))
    else:
        print(Fore.RED + "  The gpus in config file should be string of gpu ids, like '4,5' or '3', means running on cards with this ID " + Style.RESET_ALL)
        return

    print(gpu_count)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = config['gpus']
    gpus = config['gpus']
    original_log_dir = config['log_dir']
    flag = config['flag']

    """
    这里暂时没用flag区分train-test，而是直接通过`forhub test`或者`forhub train`来区分
    """
    # flag should be "train" or "test"
    if flag not in ["train", "test"]:
        print(Fore.RED + "  The flag in config file should be 'train' or 'test'" + Style.RESET_ALL)
        return

    # Get unique log_dir with increment suffix if it already exists
    base_dir = get_unique_log_dir(original_log_dir)

    # If log_dir was modified, update the config and YAML file
    if base_dir != original_log_dir:
        print(Fore.YELLOW + f"  Log directory '{original_log_dir}' already exists." + Style.RESET_ALL)
        print(Fore.YELLOW + f"  Using new log directory: '{base_dir}'" + Style.RESET_ALL)
        config['log_dir'] = base_dir
        # Update the YAML file with the new log_dir
        try:
            with open(yaml_path, 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(Fore.RED + f"  Error updating YAML file with new log_dir: {e}" + Style.RESET_ALL)
            return

    # mkdir in base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    cmd = cmd = [
    "torchrun",
    "--standalone",
    "--nnodes=1",
    f"--nproc_per_node={gpu_count}",
    script_path,
    "--config",
    yaml_path
]
    print(cmd)
    # output error file and log file
    error_file = os.path.join(base_dir, 'error.log')
    log_file = os.path.join(base_dir, 'logs.log')
    # Excute with subprocess
    # NOTE: DO NOT USE shell=True to avoid security risk
    # STD out to log file
    # STD err to error file
    # 用绿色告诉用户，script_path对应的basedir脚本已经开始执行，并告诉用户所有输出日志保存到的路径：
    print(Fore.GREEN + f"  The script {os.path.basename(script_path)} is running...\n    All logs are saved in: {log_file}\n    All error report are saved in: {error_file}" + Style.RESET_ALL)
    print()
    with open(log_file, 'a') as log_f, open(error_file, 'a') as error_f:
        process = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=error_f,
            shell=False
        )
    sys.exit(process.returncode)

def cli_train(yaml_path):
    """
    print("This is cli for training.")
    # TODO
    """
    print("This is cli for training.")
    # TODO
    print(train_script.__file__)
    print(yaml_path)
    excute_script(yaml_path, train_script.__file__)

def cli_test(yaml_path):
    """
    print("This is cli for testing.")
    # TODO
    """
    print("This is cli for testing.")
    # TODO
    print(test_script.__file__)
    print(yaml_path)
    excute_script(yaml_path, test_script.__file__)