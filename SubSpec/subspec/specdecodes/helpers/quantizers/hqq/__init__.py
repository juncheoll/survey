import logging
import importlib.util
import os

from hqq.core.quantize import *
from hqq.core.peft import HQQLinearLoRA
from hqq.utils.patching import prepare_for_inference
from .hf.base import AutoHQQHFModel

# Check for the kernel library (replace 'kernel_library' with the actual module name)
if importlib.util.find_spec("gemlite") is None:
    raise ImportError(
        "The 'gemlite' kernel is required for optimized performance. "
        "Please install it using 'pip install git+https://github.com/mobiusml/gemlite/'."
    )

class HqqQuantizer:
    @classmethod
    def quantize_model(cls, model, quant_config, compute_dtype, device):
        logging.info("Quantizing model with HqqQuantizer")

        def phase(message):
            logging.info(f"[HqqQuantizer] {message}")

        def maybe_patch_gemlite_cpu_packing():
            mode = (os.environ.get("SUBSPEC_GEMLITE_CPU_PACK") or "auto").strip().lower()
            if mode in {"0", "false", "no", "off", "disable", "disabled"}:
                return

            enable = mode in {"1", "true", "yes", "y", "on", "enable", "enabled"}
            if mode == "auto":
                try:
                    import torch

                    device_name = torch.cuda.get_device_properties(device).name.lower()
                    enable = "4090" in device_name
                except Exception:
                    enable = False

            if not enable:
                return

            try:
                import gemlite.bitpack as gemlite_bitpack  # type: ignore
                import gemlite.core as gemlite_core  # type: ignore

                def pack_weights_over_cols_cpu_then_to_device(W_q, W_nbits, packing_bitwidth, transpose):
                    out_device = W_q.device
                    packed, elements_per_sample = gemlite_bitpack.pack_weights_over_cols_torch(
                        W_q.detach().cpu(),
                        W_nbits,
                        packing_bitwidth,
                        transpose,
                    )
                    return packed.to(out_device, non_blocking=True), elements_per_sample

                gemlite_core.pack_weights_over_cols_triton = pack_weights_over_cols_cpu_then_to_device
                gemlite_bitpack.pack_weights_over_cols_triton = pack_weights_over_cols_cpu_then_to_device
                logging.warning(
                    "GemLite CUDA bitpacking is disabled; using CPU bitpacking and moving packed weights back to GPU."
                )
            except Exception as e:
                logging.warning(f"Failed to patch GemLite CPU bitpacking fallback: {e}")

        def maybe_force_gemlite_matmul_type():
            mode = (os.environ.get("SUBSPEC_GEMLITE_MATMUL_TYPE") or "auto").strip().upper()
            if mode in {"0", "FALSE", "NO", "OFF", "DISABLE", "DISABLED", "DEFAULT"}:
                return

            if mode == "AUTO":
                try:
                    import torch

                    device_name = torch.cuda.get_device_properties(device).name.lower()
                except Exception:
                    device_name = ""

                # GemLite's default 4-bit M=1 path is GEMV_REVSPLITK. On RTX 4090 this can
                # segfault inside Triton's code generator, so use the splitK GEMV kernel.
                if "4090" not in device_name:
                    return
                mode = "GEMV_SPLITK"

            valid_modes = {"GEMV", "GEMV_REVSPLITK", "GEMV_SPLITK", "GEMM_SPLITK", "GEMM"}
            if mode not in valid_modes:
                logging.warning(f"Unsupported SUBSPEC_GEMLITE_MATMUL_TYPE={mode}; leaving GemLite auto mode.")
                return

            try:
                import types

                patched = 0
                for module in model.modules():
                    if not hasattr(module, "forward_manual"):
                        continue
                    if not module.__class__.__module__.startswith("gemlite"):
                        continue

                    def _forward(self, x, _matmul_type=mode):
                        return self.forward_manual(x, matmul_type=_matmul_type)

                    module.forward = types.MethodType(_forward, module)
                    patched += 1

                if patched:
                    logging.warning(f"Forced GemLite matmul type to {mode} for {patched} modules.")
            except Exception as e:
                logging.warning(f"Failed to force GemLite matmul type '{mode}': {e}")

        # Optional GemLite tuning knobs (no-op unless env vars are set).
        # These primarily reduce warmup/autotune overhead and can improve steady-state
        # performance depending on GPU + shapes.
        gemlite_cfg = os.environ.get("SUBSPEC_GEMLITE_CONFIG") or os.environ.get("GEMLITE_CONFIG")
        gemlite_autotune = os.environ.get("SUBSPEC_GEMLITE_AUTOTUNE") or os.environ.get("GEMLITE_AUTOTUNE")
        gemlite_packing = os.environ.get("SUBSPEC_GEMLITE_PACKING_BITWIDTH")
        gemlite_kernel_caching = os.environ.get("SUBSPEC_GEMLITE_KERNEL_CACHING")
        gemlite_reset_config = os.environ.get("SUBSPEC_GEMLITE_RESET_CONFIG")
        if any([gemlite_cfg, gemlite_autotune, gemlite_packing, gemlite_kernel_caching, gemlite_reset_config]):
            try:
                import gemlite  # type: ignore

                if gemlite_reset_config is not None:
                    try:
                        val = str(gemlite_reset_config).strip().lower()
                        if val in {"1", "true", "yes", "y", "on"}:
                            gemlite.reset_config()
                            logging.info("GemLite cached kernel config reset.")
                    except Exception as e:
                        logging.warning(f"Failed to reset GemLite cached kernel config: {e}")

                if gemlite_autotune:
                    try:
                        gemlite.set_autotune(str(gemlite_autotune))
                        logging.info(f"GemLite autotune mode set to: {gemlite_autotune}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite autotune mode '{gemlite_autotune}': {e}")

                if gemlite_packing:
                    try:
                        gemlite.set_packing_bitwidth(int(gemlite_packing))
                        logging.info(f"GemLite packing bitwidth set to: {gemlite_packing}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite packing bitwidth '{gemlite_packing}': {e}")

                if gemlite_kernel_caching is not None:
                    try:
                        val = str(gemlite_kernel_caching).strip().lower()
                        enabled = val in {"1", "true", "yes", "y", "on"}
                        gemlite.set_kernel_caching(enabled)
                        logging.info(f"GemLite kernel caching set to: {enabled}")
                    except Exception as e:
                        logging.warning(f"Failed to set GemLite kernel caching: {e}")

                if gemlite_cfg:
                    try:
                        gemlite.load_config(str(gemlite_cfg))
                        logging.info(f"Loaded GemLite config: {gemlite_cfg}")
                    except Exception as e:
                        logging.warning(f"Failed to load GemLite config '{gemlite_cfg}': {e}")
            except Exception as e:
                logging.warning(f"GemLite tuning requested but gemlite import/config failed: {e}")

        phase("HQQ quantization start")
        AutoHQQHFModel.quantize_model(model, quant_config, compute_dtype=compute_dtype, device=device)
        phase("HQQ quantization done")
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        phase("HQQLinear backend set to PYTORCH")

        # By default, HQQ's gemlite backend patches HQQLinear -> GemLite A16Wn (FP16 activations, Wn weights).
        # GemLite activation quant (e.g. A8Wn_dynamic) is not wired through HQQ's backend in gemlite==0.4.6,
        # so we provide an opt-in path here.
        #
        # Supported values:
        # - SUBSPEC_GEMLITE_ACTIVATIONS=fp16 (default): weight-only, uses prepare_for_inference(..., backend="gemlite")
        # - SUBSPEC_GEMLITE_ACTIVATIONS=fp8: FP8 dynamic activations + Wn weights via gemlite.helper.A8Wn_dynamic
        hqq_backend = (os.environ.get("SUBSPEC_HQQ_BACKEND") or "gemlite").strip().lower()
        if hqq_backend in {"pytorch", "torch"}:
            logging.warning("Using HQQ PyTorch backend; GemLite inference patching is disabled.")
            return
        if hqq_backend != "gemlite":
            logging.warning(f"Unsupported SUBSPEC_HQQ_BACKEND={hqq_backend}; falling back to GemLite.")

        maybe_patch_gemlite_cpu_packing()

        act_mode = (os.environ.get("SUBSPEC_GEMLITE_ACTIVATIONS") or "fp16").strip().lower()
        if act_mode in {"fp8", "a8", "a8wn"}:
            try:
                from gemlite.helper import A8Wn_dynamic

                use_fp8e5 = (os.environ.get("SUBSPEC_GEMLITE_FP8_FORMAT") or "e4m3").strip().lower() in {
                    "e5m2",
                    "fp8e5",
                    "e5",
                }
                post_scale = str(os.environ.get("SUBSPEC_GEMLITE_POST_SCALE") or "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }

                processor = A8Wn_dynamic(device=device, post_scale=post_scale, use_fp8e5=use_fp8e5)

                def _patch_module(mod):
                    for child_name, child in list(mod._modules.items()):
                        if child is None:
                            continue

                        if isinstance(child, HQQLinear):
                            mod._modules[child_name] = processor.from_hqqlinear(child)
                            continue

                        if isinstance(child, HQQLinearLoRA):
                            # Keep LoRA wrapper, replace underlying quantized linear.
                            if isinstance(child.linear_layer, HQQLinear):
                                child.linear_layer = processor.from_hqqlinear(child.linear_layer)
                            continue

                        _patch_module(child)

                _patch_module(model)
                logging.info("Patched HQQLinear layers with GemLite A8Wn_dynamic (FP8 dynamic activations).")
                phase("GemLite FP8 patch done")
                return
            except Exception as e:
                logging.warning(
                    f"SUBSPEC_GEMLITE_ACTIVATIONS={act_mode} requested but GemLite FP8 patching failed; falling back to A16Wn weight-only. Error: {e}"
                )

        phase("GemLite prepare_for_inference start")
        prepare_for_inference(model, backend="gemlite")
        phase("GemLite prepare_for_inference done")
        maybe_force_gemlite_matmul_type()
