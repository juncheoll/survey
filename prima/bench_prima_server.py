#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url, timeout):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_until_ready(base_url, startup_timeout, request_timeout):
    deadline = time.perf_counter() + startup_timeout
    health_url = base_url.rstrip("/") + "/health"
    last_error = None

    while time.perf_counter() < deadline:
        try:
            data = get_json(health_url, request_timeout)
            if data.get("status") == "ok":
                return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code != 503:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)

        time.sleep(1.0)

    raise TimeoutError(f"server did not become ready within {startup_timeout:.0f}s; last error: {last_error}")


def tokenize(base_url, prompt, timeout):
    data = post_json(
        base_url.rstrip("/") + "/tokenize",
        {"content": prompt, "add_special": False},
        timeout,
    )
    tokens = data.get("tokens", [])
    if tokens and isinstance(tokens[0], dict):
        return [t["id"] for t in tokens]
    return tokens


def detokenize(base_url, tokens, timeout):
    data = post_json(
        base_url.rstrip("/") + "/detokenize",
        {"tokens": tokens},
        timeout,
    )
    return data.get("content", "")


def make_prompt(base_url, target_tokens, timeout):
    seed = (
        "This is a throughput benchmark prompt for PRIMA distributed inference. "
        "The content is intentionally plain and repeatable. "
    )
    prompt = seed
    while len(tokenize(base_url, prompt, timeout)) < target_tokens:
        prompt += seed

    tokens = tokenize(base_url, prompt, timeout)[:target_tokens]
    prompt = detokenize(base_url, tokens, timeout)
    actual = len(tokenize(base_url, prompt, timeout))
    return prompt, actual


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def run_one(args, prompt, index):
    payload = {
        "prompt": prompt,
        "n_predict": args.output_tokens,
        "temperature": args.temperature,
        "cache_prompt": False,
        "stream": False,
    }
    if args.seed is not None:
        payload["seed"] = args.seed + index

    start = time.perf_counter()
    data = post_json(args.url.rstrip("/") + "/completion", payload, args.timeout)
    elapsed = time.perf_counter() - start

    timings = data.get("timings", {})
    prompt_tokens = int(timings.get("prompt_n", data.get("tokens_evaluated", 0)) or 0)
    completion_tokens = int(timings.get("predicted_n", data.get("tokens_predicted", 0)) or 0)
    stopped_limit = bool(data.get("stopped_limit", False))

    return {
        "ok": True,
        "elapsed": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "stopped_limit": stopped_limit,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark a running PRIMA llama-server with concurrent requests.")
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="rank 0 llama-server base URL")
    parser.add_argument("--requests", type=int, default=32, help="total requests")
    parser.add_argument("--concurrency", type=int, default=4, help="concurrent requests")
    parser.add_argument("--input-tokens", type=int, default=512, help="target prompt tokens before server formatting")
    parser.add_argument("--output-tokens", type=int, default=128, help="n_predict per request")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=600.0, help="seconds to wait for /health before benchmarking")
    args = parser.parse_args()

    wait_until_ready(args.url, args.startup_timeout, min(args.timeout, 30.0))

    prompt, actual_prompt_tokens = make_prompt(args.url, args.input_tokens, args.timeout)
    print(
        f"server={args.url} requests={args.requests} concurrency={args.concurrency} "
        f"input_tokens={actual_prompt_tokens} output_tokens={args.output_tokens}"
    )

    results = []
    failures = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run_one, args, prompt, i) for i in range(args.requests)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                failures.append(str(exc))
    wall = time.perf_counter() - start

    prompt_total = sum(r["prompt_tokens"] for r in results)
    completion_total = sum(r["completion_tokens"] for r in results)
    total_tokens = prompt_total + completion_total
    latencies = [r["elapsed"] for r in results]
    truncated = sum(1 for r in results if r["stopped_limit"])

    print("")
    print(f"completed_requests: {len(results)}")
    print(f"failed_requests:    {len(failures)}")
    print(f"wall_time_s:        {wall:.3f}")
    print(f"request_throughput: {len(results) / wall:.3f} req/s")
    print(f"prompt_tokens:      {prompt_total} ({prompt_total / wall:.3f} tok/s)")
    print(f"completion_tokens:  {completion_total} ({completion_total / wall:.3f} tok/s)")
    print(f"total_tokens:       {total_tokens} ({total_tokens / wall:.3f} tok/s)")
    print(f"truncated_outputs:  {truncated}")
    if latencies:
        print(f"latency_avg_s:      {statistics.mean(latencies):.3f}")
        print(f"latency_p50_s:      {percentile(latencies, 50):.3f}")
        print(f"latency_p95_s:      {percentile(latencies, 95):.3f}")

    if failures:
        print("")
        print("first_failure:")
        print(failures[0])


if __name__ == "__main__":
    main()
