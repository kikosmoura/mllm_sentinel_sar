"""Carregamento, preparacao e inferencia compartilhados para MLLMs abertas."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .data_utils import atomic_write_json
from .prompts import CLASSIFICATION_PROMPT


DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class MLLMModelError(RuntimeError):
    """Erro no carregamento ou uso da MLLM."""


@dataclass
class LoadedMLLM:
    model: Any
    processor: Any
    model_id: str
    requested_revision: str | None
    resolved_revision: str | None
    adapter_path: str | None
    device: str
    dtype: str
    do_image_splitting: bool


def build_messages(
    image: Image.Image,
    assistant_label: str | None = None,
    prompt: str = CLASSIFICATION_PROMPT,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if assistant_label is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_label}],
            }
        )
    return messages


def prepare_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise MLLMModelError(f"nao foi possivel abrir a imagem {path}: {exc}") from exc


def parse_predicted_class(raw_response: str) -> str | None:
    """Aceita somente um rotulo isolado, com pequenas variacoes tipograficas."""
    normalized = raw_response.strip()
    normalized = re.sub(r"^```(?:text)?\s*|\s*```$", "", normalized, flags=re.I)
    normalized = normalized.strip().strip("`*_\"'")
    normalized = re.sub(r"[.!?,;:]+$", "", normalized).strip().upper()
    normalized = re.sub(r"[\s-]+", "_", normalized)
    if normalized in {"WATER", "NON_WATER"}:
        return normalized
    return None


def _preferred_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_mllm(
    model_id: str,
    revision: str | None = None,
    adapter_path: Path | None = None,
    use_4bit: bool = False,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    trainable_adapter: bool = False,
    do_image_splitting: bool = False,
) -> LoadedMLLM:
    if use_4bit and not torch.cuda.is_available():
        raise MLLMModelError(
            "quantizacao 4-bit requer CUDA neste pipeline; execute sem "
            "--use-4bit para o fallback LoRA"
        )
    dtype = _preferred_dtype()
    common_kwargs: dict[str, object] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    processor = AutoProcessor.from_pretrained(model_id, **common_kwargs)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and hasattr(image_processor, "do_image_splitting"):
        image_processor.do_image_splitting = do_image_splitting
    elif do_image_splitting:
        raise MLLMModelError(
            "--do-image-splitting foi solicitado, mas o processor nao suporta a opcao"
        )

    model_kwargs: dict[str, object] = dict(common_kwargs)
    model_kwargs["dtype"] = dtype
    model_kwargs["low_cpu_mem_usage"] = True
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise MLLMModelError(
                "transformers sem BitsAndBytesConfig; QLoRA indisponivel"
            ) from exc
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise MLLMModelError(
                "bitsandbytes nao instalado; instale-o ou execute sem --use-4bit"
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    if not torch.cuda.is_available() and not use_4bit:
        model.to("cpu")

    adapter_string: str | None = None
    if adapter_path is not None:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise MLLMModelError(
                "peft e necessario para carregar o adapter"
            ) from exc
        adapter_path = adapter_path.resolve()
        if not adapter_path.is_dir():
            raise MLLMModelError(f"adapter inexistente: {adapter_path}")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        configured_base = str(peft_config.base_model_name_or_path)
        if configured_base != model_id:
            raise MLLMModelError(
                f"adapter foi treinado sobre {configured_base!r}, nao {model_id!r}"
            )
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=trainable_adapter,
        )
        adapter_string = str(adapter_path)

    tokenizer = getattr(processor, "tokenizer", None)
    generation_config = getattr(model, "generation_config", None)
    if tokenizer is not None and generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    base_config = getattr(model, "config", None)
    resolved_revision = getattr(base_config, "_commit_hash", None)
    parameter = next(model.parameters())
    return LoadedMLLM(
        model=model,
        processor=processor,
        model_id=model_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        adapter_path=adapter_string,
        device=str(parameter.device),
        dtype=str(parameter.dtype),
        do_image_splitting=do_image_splitting,
    )


def prepare_inference_inputs(loaded: LoadedMLLM, image: Image.Image) -> Any:
    messages = build_messages(image)
    try:
        inputs = loaded.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except (TypeError, ValueError, NotImplementedError):
        placeholder_messages = build_messages(image)
        prompt = loaded.processor.apply_chat_template(
            placeholder_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = loaded.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )
    target_device = next(loaded.model.parameters()).device
    return inputs.to(target_device)


def generate_response(
    loaded: LoadedMLLM,
    image_path: Path,
    max_new_tokens: int = 8,
) -> str:
    image = prepare_image(image_path)
    inputs = prepare_inference_inputs(loaded, image)
    input_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = loaded.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    if getattr(loaded.model.config, "is_encoder_decoder", False):
        generated_only = generated
    else:
        generated_only = generated[:, input_length:]
    decoder = getattr(loaded.processor, "batch_decode", None)
    if decoder is None:
        decoder = loaded.processor.tokenizer.batch_decode
    return decoder(generated_only, skip_special_tokens=True)[0].strip()


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "Pillow",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_information() -> dict[str, object]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "vram_bytes": int(properties.total_memory),
                }
            )
    try:
        ram_bytes = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        ram_bytes = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram_bytes,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": cuda_devices,
    }


def update_environment_file(
    path: Path,
    run_name: str,
    run_payload: dict[str, object],
) -> None:
    payload: dict[str, object]
    if path.is_file():
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
    else:
        payload = {}
    payload["hardware"] = hardware_information()
    payload["library_versions"] = package_versions()
    runs = payload.setdefault("runs", {})
    if not isinstance(runs, dict):
        runs = {}
        payload["runs"] = runs
    run_payload = dict(run_payload)
    run_payload["recorded_unix_time"] = time.time()
    runs[run_name] = run_payload
    atomic_write_json(path, payload)
