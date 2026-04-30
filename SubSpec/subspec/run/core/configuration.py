from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import torch
from specdecodes.models.utils.utils import DraftParams

@dataclass
class AppConfig:
    # Base configurations
    method: str = "classic_sd"
    vram_limit_gb: Optional[int] = None
    seed: int = 0
    device: str = "cuda:0"
    dtype: torch.dtype = torch.float16
    
    # Model paths
    llm_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    draft_model_path: Optional[str] = None
    model_cache_dir: Optional[str] = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    
    # Generation parameters
    max_length: int = 2048
    max_new_tokens: Optional[int] = None
    test_input_tokens: Optional[int] = None
    test_input_token_text: str = " hello"
    test_prompt: Optional[str] = None
    ignore_eos: bool = False
    do_sample: bool = False
    temperature: float = 0.0
    
    # Generator-specific configurations
    generator_kwargs: Dict[str, Any] = field(default_factory=dict)
    draft_params: Optional[DraftParams] = None
    
    # Recipe
    recipe: Any = None
    cpu_offload_gb: Optional[int] = None
    
    # Additional configurations
    cache_implementation: str = "dynamic"
    warmup_iter: int = 0
    compile_mode: Optional[str] = None
    
    # Profiling
    generator_profiling: bool = True
    profiling_verbose: bool = True
    sync_token_timing: bool = True
    print_time: bool = True
    print_message: bool = True
    
    # Benchmarking/logging
    out_dir: Optional[str] = None
    log_dir: str = "experiments"

    # Settings snapshot (resolved config + CLI context)
    config_path: Optional[str] = None
    settings_snapshot: Optional[Dict[str, Any]] = None

    # Research toggles (set via YAML/CLI)
    detailed_analysis: bool = False
    nvtx_profiling: bool = False
    nsys_output: str = "nsight_report"

    def update(self, new_config: Dict[str, Any]):
        for key, value in new_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
