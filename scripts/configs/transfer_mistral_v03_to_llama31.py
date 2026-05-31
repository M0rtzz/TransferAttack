import os

os.sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.template import get_config as default_config


def _repo_model_path(*parts):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(repo_root, "Models", *parts)


def get_config():
    config = default_config()

    source_model_path = os.environ.get(
        "SOURCE_MODEL_PATH",
        _repo_model_path("Mistral-7B-Instruct-v0.3"),
    )
    target_model_path = os.environ.get(
        "TARGET_MODEL_PATH",
        _repo_model_path("Meta-Llama-3.1-8B-Instruct"),
    )
    single_gpu_search = os.environ.get("SINGLE_GPU_SEARCH", "").lower() in {
        "1",
        "true",
        "yes",
    }

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    if single_gpu_search:
        config.tokenizer_paths = [source_model_path]
        config.tokenizer_kwargs = [{"use_fast": False}]
        config.model_paths = [source_model_path]
        config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
        config.conversation_templates = ["mistral"]
        config.devices = [os.environ.get("SOURCE_DEVICE", "cuda:0")]
    else:
        config.tokenizer_paths = [
            source_model_path,
            target_model_path,
        ]
        config.tokenizer_kwargs = [
            {"use_fast": False},
            {},
        ]
        config.model_paths = [
            source_model_path,
            target_model_path,
        ]
        config.model_kwargs = [
            {"low_cpu_mem_usage": True, "use_cache": False},
            {"low_cpu_mem_usage": True, "use_cache": False},
        ]
        config.conversation_templates = [
            "mistral",
            "llama-3",
        ]
        config.devices = [
            os.environ.get("SOURCE_DEVICE", "cuda:0"),
            os.environ.get("TARGET_DEVICE", "cuda:1"),
        ]

    return config
