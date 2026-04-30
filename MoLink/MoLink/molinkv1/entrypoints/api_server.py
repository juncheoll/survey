"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import asyncio
import json
import ssl
import time
from argparse import Namespace
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import vllm.envs as envs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.version import __version__ as VLLM_VERSION
from molinkv1.arg_utils import MolinkEngineArgs
from molinkv1.engine.engine import MolinkEngine
logger = init_logger("vllm.entrypoints.api_server")

app = FastAPI()
engine = None

_OPENAI_IGNORED_FIELDS = {
    "model",
    "prompt",
    "messages",
    "stream",
    "stream_options",
    "suffix",
    "user",
    "echo",
    "metadata",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
}

_SAMPLING_FIELDS = {
    "n",
    "best_of",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "stop_token_ids",
    "bad_words",
    "include_stop_str_in_output",
    "ignore_eos",
    "max_tokens",
    "min_tokens",
    "logprobs",
    "prompt_logprobs",
    "detokenize",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "truncate_prompt_tokens",
    "structured_outputs",
    "guided_decoding",
    "logit_bias",
    "allowed_token_ids",
}


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    return await _generate(request_dict, raw_request=request)


@app.get("/v1/models")
async def show_available_models() -> JSONResponse:
    model_name = _get_served_model_name()
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "vllm",
                }
            ],
        }
    )


@app.get("/version")
async def show_version() -> JSONResponse:
    return JSONResponse({"version": VLLM_VERSION})


@app.post("/v1/completions")
async def create_completion(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.get("prompt")
    if prompt is None:
        return _openai_error("Missing required field: prompt", status_code=400)

    prompts = prompt if isinstance(prompt, list) else [prompt]
    if not all(isinstance(item, str) for item in prompts):
        return _openai_error("prompt must be a string or a list of strings", 400)

    stream = bool(request_dict.get("stream", False))
    model_name = request_dict.get("model") or _get_served_model_name()
    sampling_params = _make_sampling_params(request_dict, stream=stream)
    request_id = f"cmpl-{random_uuid()}"

    assert engine is not None
    if stream:
        return StreamingResponse(
            _completion_stream(prompts[0], sampling_params, request_id, model_name),
            media_type="text/event-stream",
        )

    created = int(time.time())
    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0

    for prompt_index, prompt_text in enumerate(prompts):
        prompt_request_id = (
            request_id if len(prompts) == 1 else f"{request_id}-{prompt_index}"
        )
        final_output = await _run_generation(
            prompt_text, sampling_params, prompt_request_id
        )
        if final_output.prompt_token_ids:
            prompt_tokens += len(final_output.prompt_token_ids)
        for output in final_output.outputs:
            completion_tokens += len(output.token_ids)
            choices.append(
                {
                    "index": len(choices),
                    "text": prompt_text + output.text
                    if request_dict.get("echo", False)
                    else output.text,
                    "logprobs": None,
                    "finish_reason": output.finish_reason or "stop",
                }
            )

    return JSONResponse(
        {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": choices,
            "usage": _usage(prompt_tokens, completion_tokens),
        }
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request) -> Response:
    request_dict = await request.json()
    messages = request_dict.get("messages")
    if not isinstance(messages, list) or not messages:
        return _openai_error("messages must be a non-empty list", status_code=400)

    stream = bool(request_dict.get("stream", False))
    model_name = request_dict.get("model") or _get_served_model_name()
    sampling_params = _make_sampling_params(request_dict, stream=stream)
    prompt = await _messages_to_prompt(messages)
    request_id = f"chatcmpl-{random_uuid()}"

    assert engine is not None
    if stream:
        return StreamingResponse(
            _chat_stream(prompt, sampling_params, request_id, model_name),
            media_type="text/event-stream",
        )

    created = int(time.time())
    final_output = await _run_generation(prompt, sampling_params, request_id)
    prompt_tokens = len(final_output.prompt_token_ids or [])
    completion_tokens = sum(len(output.token_ids) for output in final_output.outputs)
    choices = [
        {
            "index": output.index,
            "message": {"role": "assistant", "content": output.text},
            "logprobs": None,
            "finish_reason": output.finish_reason or "stop",
        }
        for output in final_output.outputs
    ]

    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": choices,
            "usage": _usage(prompt_tokens, completion_tokens),
        }
    )


@with_cancellation
async def _generate(request_dict: dict, raw_request: Request) -> Response:
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            assert prompt is not None
            text_outputs = [prompt + output.text for output in request_output.outputs]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\n").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        return Response(status_code=499)

    assert final_output is not None
    prompt = final_output.prompt
    assert prompt is not None
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


async def _run_generation(
    prompt: str, sampling_params: SamplingParams, request_id: str
) -> Any:
    final_output = None
    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)
    try:
        async for request_output in results_generator:
            final_output = request_output
    except asyncio.CancelledError:
        raise

    assert final_output is not None
    return final_output


def _make_sampling_params(
    request_dict: dict[str, Any], *, stream: bool
) -> SamplingParams:
    params = dict(request_dict)
    if "max_completion_tokens" in params and "max_tokens" not in params:
        params["max_tokens"] = params["max_completion_tokens"]

    sampling_kwargs = {
        key: value
        for key, value in params.items()
        if key in _SAMPLING_FIELDS and key not in _OPENAI_IGNORED_FIELDS
    }
    sampling_kwargs["output_kind"] = (
        RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY
    )
    return SamplingParams.from_optional(**sampling_kwargs)


def _get_served_model_name() -> str:
    if engine is not None:
        model_config = getattr(engine, "model_config", None)
        served_model_name = getattr(model_config, "served_model_name", None)
        if isinstance(served_model_name, str):
            return served_model_name
        if served_model_name:
            return served_model_name[0]
        model = getattr(model_config, "model", None)
        if model:
            return model
    return "molink-model"


def _openai_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
        status_code=status_code,
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return "" if content is None else str(content)


async def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    assert engine is not None
    try:
        tokenizer = await engine.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception as exc:
        logger.warning("Falling back to simple chat prompt formatting: %s", exc)

    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = _message_content_to_text(message.get("content"))
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


async def _completion_stream(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str,
) -> AsyncGenerator[bytes, None]:
    assert engine is not None
    created = int(time.time())
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        for output in request_output.outputs:
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": output.index,
                        "text": output.text,
                        "logprobs": None,
                        "finish_reason": output.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


async def _chat_stream(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str,
) -> AsyncGenerator[bytes, None]:
    assert engine is not None
    created = int(time.time())
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        for output in request_output.outputs:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": output.index,
                        "delta": {"content": output.text},
                        "logprobs": None,
                        "finish_reason": output.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


def build_app(args: Namespace) -> FastAPI:
    global app

    app.root_path = args.root_path
    return app


async def init_app(
    args: Namespace,
    llm_engine: MolinkEngine | None = None,
) -> FastAPI:
    app = build_app(args)

    global engine

    engine_args = MolinkEngineArgs.from_cli_args(args)
    engine = (
        llm_engine
        if llm_engine is not None
        else MolinkEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER
        )
    )
    app.state.engine_client = engine
    return app


async def run_server(
    args: Namespace, llm_engine: MolinkEngine | None = None, **uvicorn_kwargs: Any
) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()

    app = await init_app(args, llm_engine)
    assert engine is not None

    shutdown_task = await serve_http(
        app,
        sock=None,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        **uvicorn_kwargs,
    )

    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=parser.check_port, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--enable-ssl-refresh",
        action="store_true",
        default=False,
        help="Refresh SSL Context when SSL certificate files change",
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    parser.add_argument("--log-level", type=str, default="debug")
    parser = MolinkEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
