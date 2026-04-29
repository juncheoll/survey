# Credit https://github.com/sayakpaul
from save_model_utils import get_model, validate_arguments, parse_arguments
from torch_save_utils import load_io_ops, _test_ds_fast_save, test_save
import safetensors.torch
import os
import time
import torch

def test_sft_save(file, buffer, args):
    st = time.time()
    safetensors.torch.save_file(filename=file, tensors=buffer)
    return time.time() - st

def main():
    print(
        f'Performance test of torch.save() integration of fast model checkpointing.'
    )
    print(f'torch version = {torch.__version__}')
    torch.manual_seed(42)

    args = parse_arguments()
    if not validate_arguments(args):
        quit()
    load_io_ops(args)
    model, tokenizer, model_name, ckpt_name = get_model(args.model)

    inputs = tokenizer("I am good", return_tensors="pt").to("cuda")

    if args.half:
        model = model.half()
    if args.gpu:
        model = model.to("cuda")
    
    with torch.no_grad():
        model.eval()
        pre_logits = model(**inputs).logits
     
    if not args.safetensors:
        file = os.path.join(args.folder, f'{ckpt_name}.pt')
    else:
        file = os.path.join(args.folder, f'{ckpt_name}.safetensors')
    if os.path.exists(file):
        os.remove(file)
    if not args.regular_torch_save and not args.safetensors:
        write_sec = _test_ds_fast_save(file, model.state_dict(), args, False)
    elif args.regular_torch_save:
        write_sec = test_save(file, model.state_dict(), args)
    else:
        write_sec = test_sft_save(file, model.state_dict(), args)
    ckpt_size = os.path.getsize(file)
    gb_size = ckpt_size / (1024**3)
    gb_per_sec = gb_size / write_sec
    print(
        f'{gb_size:5.2f} GB, {write_sec:5.2f} secs, {gb_per_sec:5.2f} GB/s'
    )
    st = time.time()
    if args.safetensors:
        loaded_sd = safetensors.torch.load_file(file, device="cuda")
    else:
        loaded_sd = torch.load(file, weights_only=True, map_location="cuda")
    load_sec = time.time() - st
    print(f"Loaded in {load_sec:5.2f} seconds.")
    model.load_state_dict(loaded_sd)
    with torch.no_grad():
        model.eval()
        post_logits = model(**inputs).logits
    
    assert torch.allclose(pre_logits, post_logits, atol=1e-3, rtol=1e-3)
    os.remove(file)


if __name__ == "__main__":
    main()