# config.py

import os

# import gpt4all
import openai
import questionary
import yaml

from consts import MODEL_TYPES, OPENROUTER_API_BASE # Import OPENROUTER_API_BASE

config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".config.yaml")


def get_config():
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    return config


def save_config(config):
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def api_key_is_invalid(api_key, model_type): # Add model_type parameter
    if not api_key:
        return True
    if model_type == MODEL_TYPES["OPENROUTER"]: # Add check for OpenRouter
        # For OpenRouter, a simple check for non-empty key is often sufficient
        return False
    try:
        openai.api_key = api_key
        # openai.Engine.list() # OpenRouter doesn't typically expose this
    except Exception:
        return True
    return False


def get_gpt_models(openai_instance): # Renamed parameter to avoid conflict
    try:
        # OpenRouter models are not listed via openai.Model.list() in the same way
        # You would typically hardcode or fetch a list of available models from OpenRouter's documentation
        # For simplicity, if using OpenRouter, you might skip this step or
        # present common OpenRouter models to the user.
        # For this example, we will return an empty list if model_type is OPENROUTER
        if openai_instance.api_base == OPENROUTER_API_BASE:
            return ["openrouter/auto", "openrouter/gpt-4-turbo", "openrouter/mistral-7b-instruct"] # Example OpenRouter models
        
        model_lst = openai_instance.Model.list()
    except Exception as e:
        print("✘ Failed to retrieve model list")
        print(e)
        return []

    return [i['id'] for i in model_lst['data'] if 'gpt' in i['id']]


def configure_model_name_openai(config):
    api_key = config.get("api_key")

    if config.get("model_type") not in [MODEL_TYPES["OPENAI"], MODEL_TYPES["OPENROUTER"]]: # Adjust check
        return

    openai.api_key = api_key
    if config.get("model_type") == MODEL_TYPES["OPENROUTER"]: # Set base URL for OpenRouter
        openai.api_base = OPENROUTER_API_BASE
    else:
        openai.api_base = "https://api.openai.com/v1" # Reset for OpenAI if needed
    
    gpt_models = get_gpt_models(openai)
    choices = [{"name": model, "value": model} for model in gpt_models]

    if not choices:
        print("ℹ No models available. Please check your API key and model type settings.")
        return

    model_name = questionary.select("🤖 Select model name:", choices).ask()

    if not model_name:
        print("✘ No model selected")
        return

    config["openai_model_name"] = model_name
    save_config(config)
    print("🤖 Model name saved!")


def remove_model_name_openai():
    config = get_config()
    config["openai_model_name"] = None
    save_config(config)


def get_and_validate_api_key(config):
    prompt = "🤖 Enter your API key: "
    api_key = input(prompt)
    while api_key_is_invalid(api_key, config.get("model_type")): # Pass model_type
        print("✘ Invalid API key")
        api_key = input(prompt)
    return api_key


def configure_api_key(config):
    if config.get("model_type") not in [MODEL_TYPES["OPENAI"], MODEL_TYPES["OPENROUTER"]]: # Adjust check
        return

    if api_key_is_invalid(config.get("api_key"), config.get("model_type")): # Pass model_type
        api_key = get_and_validate_api_key(config)
        config["api_key"] = api_key
        save_config(config)
    return


def remove_api_key():
    config = get_config()
    config["api_key"] = None
    save_config(config)


def remove_model_type():
    config = get_config()
    config["model_type"] = None
    save_config(config)


def configure_model_type(config):
    model_type = questionary.select(
        "🤖 Select model type:",
        choices=[
            {"name": "OpenAI", "value": MODEL_TYPES["OPENAI"]},
            {"name": "OpenRouter", "value": MODEL_TYPES["OPENROUTER"]}, # Add OpenRouter option
        ]
    ).ask()
    config["model_type"] = model_type
    save_config(config)


CONFIGURE_STEPS = [
    configure_model_type,
    configure_api_key,
    configure_model_name_openai,
]