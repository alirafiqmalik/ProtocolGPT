import os
import sys
import fire
import time
import logging

from config import CONFIGURE_STEPS, save_config, get_config, config_path, remove_api_key, remove_model_type, remove_model_name_local, remove_model_name_openai
from consts import DEFAULT_CONFIG
from llm import factory_llm, load_local_files
from utils import get_repo


def check_python_version():
    if sys.version_info < (3, 8, 1):
        print("🤖 Please use Python 3.8.1 or higher")
        sys.exit(1)


def setup_logging():
    logging.basicConfig(
        filename='chat_log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'
    )
    

def update_config(config):
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    return config


def configure(reset=True):
    # if reset:
    #     remove_api_key()
    #     remove_model_type()
    #     remove_model_name_local()
    #     remove_model_name_openai()
    config = get_config()
    config = update_config(config)
    for step in CONFIGURE_STEPS:
        step(config)
    save_config(config)


def chat(dir):
    # configure(False)
    config = get_config()
    logging.info(f"Configure: {config}")
    repo = get_repo()
    codebase_path = dir
    logging.info(f"Codebase: {codebase_path}")
    llm = factory_llm(codebase_path, config)
    llm.chat_loop()

def create_vectores():
    # configure(False)
    config = get_config()
    # codebases = ["feng", "openbgpd-openbsd", "openl2tp", "s2n-tls"]
    codebases = ["openbgpd-openbsd"]
    codebase_paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_code", codebase) for codebase in codebases]
    print(codebase_paths)
    for codebase in codebase_paths:
        llm = factory_llm(codebase, config)
        time.sleep(10)


def main():
    setup_logging()
    check_python_version()
    logging.info("ProtocolGPT start")
    logging.info(f"Config path: {config_path}")
    print(f"🤖 Config path: {config_path}:")
    
    try:
        fire.Fire({
            "chat": chat,
            "config": lambda: configure(True)
        })
    except KeyboardInterrupt:
        print("\n🤖 Bye!")
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
