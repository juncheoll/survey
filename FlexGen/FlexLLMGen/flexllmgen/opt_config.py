"""
The OPT model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
"""

import argparse
import dataclasses
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm


@dataclasses.dataclass(frozen=True)
class OptConfig:
    name: str = "opt-125m"
    num_hidden_layers: int = 12
    max_seq_len: int = 2048
    hidden_size: int = 768
    n_head: int = 12
    input_dim: int = 768
    ffn_embed_dim: int = 3072
    pad: int = 1
    activation_fn: str = 'relu'
    vocab_size: int = 50272
    layer_norm_eps: float = 0.00001
    pad_token_id: int = 1
    dtype: type = np.float16
    model_type: str = "opt"

    def model_bytes(self):
        h = self.input_dim
        return 	2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "llama-7b"
    num_hidden_layers: int = 32
    max_seq_len: int = 4096
    hidden_size: int = 4096
    n_head: int = 32
    input_dim: int = 4096
    ffn_embed_dim: int = 11008
    pad: int = 0
    activation_fn: str = 'silu'
    vocab_size: int = 32000
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    eos_token_id: int = 2
    dtype: type = np.float16
    model_type: str = "llama"
    rope_theta: float = 10000.0

    def model_bytes(self):
        h = self.input_dim
        return 2 * (self.num_hidden_layers * (
        # self-attention (no bias in llama)
        h * (3 * h) + h * h +
        # mlp (gate_proj, up_proj, down_proj; no bias in llama)
        h * self.ffn_embed_dim * 3 +
        # layer norm
        h * 2) +
        # input embedding, output embedding, final norm
        self.vocab_size * h * 2 + h)

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2


def get_opt_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    # Handle opt-iml-30b and opt-iml-max-30b
    if "-iml-max" in name:
        arch_name = name.replace("-iml-max", "")
    elif "-iml" in name:
        arch_name = name.replace("-iml", "")
    else:
        arch_name = name

    if arch_name == "opt-125m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=12, n_head=12,
            hidden_size=768, input_dim=768, ffn_embed_dim=768 * 4,
        )
    elif arch_name == "opt-350m":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=16,
            hidden_size=1024, input_dim=1024, ffn_embed_dim=1024 * 4,
        )
        raise NotImplementedError("Not implemented because this model "
                                  "has a different architecture")
    elif arch_name == "opt-1.3b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=2048 * 4,
        )
    elif arch_name == "opt-2.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, input_dim=2560, ffn_embed_dim=2560 * 4,
        )
    elif arch_name == "opt-6.7b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=4096 * 4,
        )
    elif arch_name == "opt-13b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=40, n_head=40,
            hidden_size=5120, input_dim=5120, ffn_embed_dim=5120 * 4,
        )
    elif arch_name == "opt-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4,
        )
    elif arch_name == "galactica-30b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=48, n_head=56,
            hidden_size=7168, input_dim=7168, ffn_embed_dim=7168 * 4, vocab_size=50000,
        )
    elif arch_name == "opt-66b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=64, n_head=72,
            hidden_size=9216, input_dim=9216, ffn_embed_dim=9216 * 4,
        )
    elif arch_name == "opt-175b":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=96, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    elif arch_name == "opt-175b-stage":
        config = OptConfig(name=name,
            max_seq_len=2048, num_hidden_layers=24, n_head=96,
            hidden_size=12288, input_dim=12288, ffn_embed_dim=12288 * 4,
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)


def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    arch_name = name

    if arch_name == "llama-2-7b-chat-hf" or arch_name == "llama-2-7b-hf" or arch_name == "llama-7b":
        config = LlamaConfig(name=name,
            max_seq_len=4096, num_hidden_layers=32, n_head=32,
            hidden_size=4096, input_dim=4096, ffn_embed_dim=11008,
        )
    elif (arch_name == "llama-2-13b-chat-hf" or
          arch_name == "llama-2-13b-hf" or arch_name == "llama-13b"):
        config = LlamaConfig(name=name,
            max_seq_len=4096, num_hidden_layers=40, n_head=40,
            hidden_size=5120, input_dim=5120, ffn_embed_dim=13824,
        )
    elif (arch_name == "llama-2-70b-chat-hf" or
          arch_name == "llama-2-70b-hf" or arch_name == "llama-70b"):
        config = LlamaConfig(name=name,
            max_seq_len=4096, num_hidden_layers=80, n_head=64,
            hidden_size=8192, input_dim=8192, ffn_embed_dim=28672,
        )
    elif arch_name == "tinyllama-1.1b-chat-v1.0" or arch_name == "tinyllama":
        config = LlamaConfig(name=name,
            max_seq_len=2048, num_hidden_layers=22, n_head=32,
            hidden_size=2048, input_dim=2048, ffn_embed_dim=5632,
            vocab_size=32000,
        )
    else:
        # Try to extract from model name as fallback
        raise ValueError(f"Invalid Llama model name: {name}")

    return dataclasses.replace(config, **kwargs)


def get_model_config(name, **kwargs):
    """Auto-detect model type and load appropriate config."""
    name_lower = name.lower()
    
    if "opt" in name_lower:
        return get_opt_config(name, **kwargs)
    elif "llama" in name_lower or "tinyllama" in name_lower:
        return get_llama_config(name, **kwargs)
    else:
        # Default to OPT config
        return get_opt_config(name, **kwargs)


def download_opt_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import OPTForCausalLM, BloomForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "galactica" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_opt_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.opt.modeling_opt.OPTPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_opt_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
    elif "galactica" in model_name:
        hf_model_name = "facebook/" + model_name

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())

            # shared embedding
            if "decoder.embed_tokens.weight" in name:
                shutil.copy(param_path, param_path.replace(
                    "decoder.embed_tokens.weight", "lm_head.weight"))


def get_llama_hf_model_name(model_name):
    if "/" in model_name:
        return model_name

    name = model_name.lower()
    if name.startswith("llama-2-"):
        parts = name.split("-")
        size = parts[2]
        is_chat = "chat" in parts
        chat = "-chat" if is_chat else ""
        return f"meta-llama/Llama-2-{size}{chat}-hf"
    elif name == "llama-7b":
        return "meta-llama/Llama-2-7b-hf"
    elif name == "llama-13b":
        return "meta-llama/Llama-2-13b-hf"
    elif name == "llama-70b":
        return "meta-llama/Llama-2-70b-hf"
    elif name == "tinyllama" or name == "tinyllama-1.1b-chat-v1.0":
        return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    raise ValueError(f"Invalid Llama model name: {model_name}")


def download_llama_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    hf_model_name = get_llama_hf_model_name(model_name)
    print(f"Load the pre-trained pytorch weights of {hf_model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    folder = snapshot_download(
        hf_model_name,
        allow_patterns=["*.bin", "*.safetensors"],
    )
    safetensor_files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    bin_files = sorted(glob.glob(os.path.join(folder, "*.bin")))
    weight_files = safetensor_files if safetensor_files else bin_files
    if not weight_files:
        raise FileNotFoundError(
            f"No .safetensors or .bin weight files found for {hf_model_name}"
        )

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    else:
        model_name = model_name.lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for weight_file in tqdm(weight_files, desc="Convert Llama format"):
        if weight_file.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as e:
                raise ImportError(
                    "safetensors is required to convert Llama safetensor weights"
                ) from e
            state = load_file(weight_file, device="cpu")
        else:
            state = torch.load(weight_file, map_location="cpu")

        for name, param in tqdm(state.items(), leave=False):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str, default="~/opt_weights")
    args = parser.parse_args()

    model_name_lower = args.model.lower()
    if "llama" in model_name_lower or "tinyllama" in model_name_lower:
        download_llama_weights(args.model, args.path)
    else:
        download_opt_weights(args.model, args.path)
