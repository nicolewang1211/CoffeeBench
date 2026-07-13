"""Local model wrapper — runs models locally without API calls.

Supports:
- Hugging Face transformers models (via transformers library)
- GGUF models (via llama-cpp-python)
- vLLM for optimized inference (optional)

Usage examples:
- "local:meta-llama/Llama-3.1-8B-Instruct" — Hugging Face model
- "local:path/to/model.gguf" — GGUF model file
- "local-vllm:meta-llama/Llama-3.1-8B-Instruct" — vLLM backend
"""

from __future__ import annotations

import json
import os
import platform
import threading
import uuid
from pathlib import Path
from typing import Any

from coffeebench.models._retry import call_with_retry
from coffeebench.models.types import ModelResponse, ToolCall, ToolSpec


class LocalModel:
    """Local model wrapper supporting multiple inference backends.

    Caches the underlying model/tokenizer objects so multiple agents using the
    same model share the same in-memory weights. Each wrapper keeps its own
    usage counters for cost/stat tracking.
    """

    DEFAULT_MAX_INPUT_TOKENS = 32_768

    # Shared cache keyed by (model_name, backend, device) to avoid loading the
    # same weights multiple times when several agents use the same model.
    _shared_cache: dict[tuple, tuple] = {}

    def __init__(
        self,
        model: str,
        backend: str = "auto",
        device: str = "auto",
        enable_thinking: bool = False,
    ):
        """Initialize local model.

        Args:
            model: Model identifier (HF model name or path to GGUF file)
            backend: Inference backend ("auto", "transformers", "gguf", "vllm")
            device: Device to run on ("auto", "cuda", "cpu", "mps")
            enable_thinking: Whether to enable chain-of-thought reasoning
        """
        self.cost = 0.0  # Local models have no API cost
        self.n_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_input_tokens = 0
        self.max_input_tokens = self.DEFAULT_MAX_INPUT_TOKENS
        self.model_name = model
        self.model = f"local:{model}"  # String identifier for agent.py compatibility
        self.enable_thinking = enable_thinking
        self.max_tokens = 4096
        self.temperature = 0.0

        # Auto-detect backend if not specified
        if backend == "auto":
            if model.endswith(".gguf") or Path(model).suffix == ".gguf":
                backend = "gguf"
            elif os.getenv("USE_VLLM", "").lower() in ("1", "true", "yes"):
                backend = "vllm"
            else:
                backend = "transformers"

        self.backend = backend
        self.device = device
        self._model_obj = None  # Actual model object (PyTorch/llama.cpp/vLLM)
        self.tokenizer = None
        self._gguf_lock = None  # Lock for shared llama.cpp context
        self._initialize_backend()

    def _initialize_backend(self):
        """Initialize the selected inference backend."""
        if self.backend == "transformers":
            self._init_transformers()
        elif self.backend == "gguf":
            self._init_gguf()
        elif self.backend == "vllm":
            self._init_vllm()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _init_transformers(self):
        """Initialize Hugging Face transformers backend."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for local models. "
                "Install with: pip install torch transformers accelerate"
            ) from e

        # Determine device
        disable_mps = os.getenv("PYTORCH_MPS_DISABLE", "").lower() in ("1", "true", "yes")
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif not disable_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                if disable_mps:
                    print("[local:transformers] MPS disabled via PYTORCH_MPS_DISABLE, using CPU")

        cache_key = (self.model_name, self.backend, self.device)
        if cache_key in LocalModel._shared_cache:
            print(f"[local:{self.backend}] Reusing cached model: {self.model_name}")
            self._model_obj, self.tokenizer = LocalModel._shared_cache[cache_key]
            print(f"[local:{self.backend}] Model loaded on {self.device}")
            return

        print(f"[local:{self.backend}] Loading model: {self.model_name}")

        # Hugging Face token (optional, for gated models and higher download limits)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        local_only = os.getenv("COFFEEBENCH_LOCAL_ONLY", "").lower() in ("1", "true", "yes")
        hf_kwargs = {"trust_remote_code": True}
        if hf_token:
            hf_kwargs["token"] = hf_token
            print("[local:transformers] Using HF_TOKEN for authentication")
        if local_only:
            hf_kwargs["local_files_only"] = True
            print("[local:transformers] LOCAL_ONLY mode: only using cached models")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **hf_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            **hf_kwargs,
            "torch_dtype": "auto",
        }

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        elif self.device == "cpu":
            model_kwargs["torch_dtype"] = torch.float32

        self._model_obj = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        if self.device == "mps":
            self._model_obj = self._model_obj.to("mps")

        LocalModel._shared_cache[cache_key] = (self._model_obj, self.tokenizer)
        print(f"[local:{self.backend}] Model loaded on {self.device}")

    def _init_gguf(self):
        """Initialize llama-cpp-python backend for GGUF models."""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for GGUF models. "
                "Install with: pip install llama-cpp-python"
            ) from e

        # Share Llama objects across agents using the same GGUF model, but keep
        # a per-model lock because llama.cpp contexts are not thread-safe.
        cache_key = (self.model_name, self.backend, self.device)
        if cache_key in LocalModel._shared_cache:
            print(f"[local:{self.backend}] Reusing cached GGUF model: {self.model_name}")
            self._model_obj, self._gguf_lock = LocalModel._shared_cache[cache_key]
            print(f"[local:{self.backend}] GGUF model loaded")
            return

        # Resolve model path: local file or HuggingFace repo_id/filename
        model_path = Path(self.model_name)
        if not model_path.exists():
            # Treat as HuggingFace path: repo_id/filename (repo_id has one slash)
            from huggingface_hub import hf_hub_download
            print(f"[local:{self.backend}] Resolving HF path: {self.model_name}")
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            local_only = os.getenv("COFFEEBENCH_LOCAL_ONLY", "").lower() in ("1", "true", "yes")
            hf_kwargs = {"token": hf_token} if hf_token else {}
            if local_only:
                hf_kwargs["local_files_only"] = True

            repo_id, filename = self.model_name.split("/", 1)
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    **hf_kwargs,
                )
            except Exception as e:
                if local_only and "Cannot find the requested files" in str(e):
                    print(f"[local:{self.backend}] Offline cache lookup failed, trying cache with a metadata check...")
                    del hf_kwargs["local_files_only"]
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        **hf_kwargs,
                    )
                else:
                    raise FileNotFoundError(
                        f"GGUF model not found and could not resolve from HuggingFace: {self.model_name}"
                    ) from e
        else:
            model_path = str(model_path)

        print(f"[local:{self.backend}] Loading GGUF model: {model_path}")

        # Cap context for GGUF (llama.cpp allocates memory for the full n_ctx)
        self.max_input_tokens = min(self.max_input_tokens, 32_768)

        # Determine n_gpu_layers based on device
        n_gpu_layers = 0
        env_layers = os.getenv("LLAMA_CPP_N_GPU_LAYERS")
        if env_layers is not None:
            n_gpu_layers = int(env_layers)
        elif self.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    n_gpu_layers = -1  # Use all layers on CUDA
            except ImportError:
                pass
            if n_gpu_layers == 0 and platform.system() == "Darwin":
                n_gpu_layers = -1  # Use all layers on Metal (Mac)

        if n_gpu_layers:
            print(f"[local:{self.backend}] Offloading {n_gpu_layers} layers to GPU")

        self._gguf_lock = threading.Lock()
        self._model_obj = Llama(
            model_path=str(model_path),
            n_ctx=self.max_input_tokens,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        LocalModel._shared_cache[cache_key] = (self._model_obj, self._gguf_lock)
        print(f"[local:{self.backend}] GGUF model loaded")

    def _init_vllm(self):
        """Initialize vLLM backend for optimized inference."""
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vllm is required for vLLM backend. "
                "Install with: pip install vllm"
            ) from e

        # Check cache first - vLLM instances are expensive to create
        cache_key = (self.model_name, self.backend, "cuda")  # vLLM always uses CUDA
        if cache_key in LocalModel._shared_cache:
            print(f"[local:{self.backend}] Reusing cached vLLM model: {self.model_name}")
            self._model_obj, self.tokenizer = LocalModel._shared_cache[cache_key]
            print(f"[local:{self.backend}] vLLM model loaded from cache")
            return

        print(f"[local:{self.backend}] Loading model with vLLM: {self.model_name}")

        # Let vLLM auto-detect max_model_len from model config unless overridden
        vllm_kwargs = {
            "model": self.model_name,
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.85,  # Use 85% for better performance
            "max_model_len": 8192,  # Reduce from 32k to save KV cache memory
        }
        
        # Enable tensor parallelism only for large models that need it
        # Small models (7B, 14B) fit on single GPU and work better without TP
        tensor_parallel_size = os.getenv("VLLM_TENSOR_PARALLEL_SIZE")
        if tensor_parallel_size:
            vllm_kwargs["tensor_parallel_size"] = int(tensor_parallel_size)
            print(f"[local:vllm] Using tensor parallelism across {tensor_parallel_size} GPUs")
        
        # GPU memory utilization override
        gpu_mem_util = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
        if gpu_mem_util:
            vllm_kwargs["gpu_memory_utilization"] = float(gpu_mem_util)
            print(f"[local:vllm] Using GPU memory utilization: {gpu_mem_util}")
        
        # Only set max_model_len if explicitly configured via env var
        max_len_override = os.getenv("VLLM_MAX_MODEL_LEN")
        if max_len_override:
            vllm_kwargs["max_model_len"] = int(max_len_override)
            print(f"[local:vllm] Using VLLM_MAX_MODEL_LEN={max_len_override}")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            vllm_kwargs["hf_token"] = hf_token
            print("[local:vllm] Using HF_TOKEN for authentication")

        self._model_obj = LLM(**vllm_kwargs)
        self.tokenizer = self._model_obj.get_tokenizer()
        
        # Cache the vLLM instance for reuse by other agents
        LocalModel._shared_cache[cache_key] = (self._model_obj, self.tokenizer)
        print(f"[local:{self.backend}] vLLM model loaded")

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages into a prompt string."""
        # Use tokenizer's chat template if available
        if self.backend == "transformers" and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback: simple formatting
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            elif role == "tool":
                prompt_parts.append(f"Tool Result: {content}\n")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def _format_tools_in_prompt(
        self, messages: list[dict], tools: list[ToolSpec] | None
    ) -> list[dict]:
        """Add tool descriptions to the system message for models without native tool support."""
        if not tools:
            return messages

        tool_descriptions = []
        for tool in tools:
            tool_desc = f"- {tool.name}: {tool.description}\n"
            tool_desc += f"  Parameters: {json.dumps(tool.input_schema, indent=2)}"
            tool_descriptions.append(tool_desc)

        tools_text = (
            "\n\nYou have access to the following tools:\n"
            + "\n".join(tool_descriptions)
            + "\n\nTo use a tool, respond with a JSON object in this format:\n"
            + '{"tool": "tool_name", "arguments": {...}}\n'
        )

        # Add tools to system message or create one
        modified_messages = []
        system_added = False
        for msg in messages:
            if msg.get("role") == "system" and not system_added:
                modified_messages.append({
                    "role": "system",
                    "content": msg.get("content", "") + tools_text,
                })
                system_added = True
            else:
                modified_messages.append(msg)

        if not system_added:
            modified_messages.insert(0, {"role": "system", "content": tools_text})

        return modified_messages

    def _generate_transformers(self, prompt: str) -> str:
        """Generate response using transformers backend."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model_obj.device) for k, v in inputs.items()}

        self.last_input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model_obj.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (exclude input)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += len(generated_tokens)

        return response

    def _generate_gguf(self, prompt: str) -> str:
        """Generate response using llama-cpp-python backend."""
        # llama.cpp contexts are not thread-safe, so serialize access when the
        # model is shared across agents.
        with self._gguf_lock:
            output = self._model_obj(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["User:", "Human:", "\n\n\n"],
            )

        response = output["choices"][0]["text"]

        # Estimate token counts (llama-cpp provides these in usage)
        usage = output.get("usage", {})
        self.last_input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += output_tokens

        return response

    def _generate_vllm(self, prompt: str) -> str:
        """Generate response using vLLM backend."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            max_tokens=self.max_tokens,
            stop=["User:", "Human:"],
        )

        outputs = self._model_obj.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        # Track token usage
        self.last_input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(outputs[0].outputs[0].token_ids)

        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += output_tokens

        return response

    def _parse_tool_call(self, response: str) -> tuple[str, list[ToolCall]]:
        """Parse tool calls from model response."""
        tool_calls = []
        content = response.strip()

        # Try to parse JSON tool call
        try:
            # Look for JSON object in response
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)

                if "tool" in data and "arguments" in data:
                    tool_calls.append(
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            name=data["tool"],
                            input=data["arguments"],
                        )
                    )
                    # Remove tool call from content
                    content = content[:start] + content[end:]
                    content = content.strip()
        except (json.JSONDecodeError, ValueError):
            pass

        return content, tool_calls

    def query(
        self,
        messages: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResponse:
        """Query the local model."""

        # Add tools to prompt if provided
        if tools:
            messages = self._format_tools_in_prompt(messages, tools)

        # Format messages into prompt
        prompt = self._format_messages(messages)

        def _do_call():
            if self.backend == "transformers":
                return self._generate_transformers(prompt)
            elif self.backend == "gguf":
                return self._generate_gguf(prompt)
            elif self.backend == "vllm":
                return self._generate_vllm(prompt)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        # Use retry logic for robustness
        response = call_with_retry(_do_call, label=f"local:{self.backend}")

        # Parse tool calls if tools were provided
        content, tool_calls = self._parse_tool_call(response) if tools else (response, [])

        self.n_calls += 1

        print(
            f"[local:{self.backend}:{self.model_name}] "
            f"in={self.last_input_tokens} out={self.total_output_tokens - (self.total_output_tokens - len(response.split()))} "
            f"cost=$0.00"
        )

        return ModelResponse(
            content=content,
            thinking="",  # Local models don't have separate thinking output
            tool_calls=tool_calls,
            stop_reason="stop",
            cost=0.0,  # No API cost for local models
            raw=None,
        )

    def get_usage_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "n_model_calls": self.n_calls,
            "model_cost": 0.0,  # Local models have no cost
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "last_input_tokens": self.last_input_tokens,
        }

    def summarize(self, instructions: str, content: str, max_tokens: int = 4096) -> str:
        """Summarize content using the local model."""
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ]

        old_max_tokens = self.max_tokens
        self.max_tokens = max_tokens

        response = self.query(messages)

        self.max_tokens = old_max_tokens

        return response.content


if __name__ == "__main__":
    # Example usage
    model = LocalModel(
        model="meta-llama/Llama-3.2-1B-Instruct",
        backend="transformers",
    )
    response = model.query([{"role": "user", "content": "Hello! How are you?"}])
    print(response.content)
    print(model.get_usage_stats())
