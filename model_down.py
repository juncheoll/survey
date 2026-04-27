import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="예: meta-llama/Llama-2-7b-hf")
    parser.add_argument("--token", type=str, default=None, help="private/gated 모델이면 HF 토큰")
    args = parser.parse_args()

    local_path = snapshot_download(
        repo_id=args.model_name,
        token=args.token,          # 공개 모델이면 없어도 됨
        repo_type="model",
        resume_download=True,
    )

    print(f"Downloaded or reused cache at: {local_path}")


if __name__ == "__main__":
    main()