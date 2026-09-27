"""Inicializacao ImageNet explicita e ajuste parcial com BatchNorm congelada."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from cnn.data import CNNError, file_sha256


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MODES = ("head-only", "layer4+fc")


def build_pretrained(cache_dir: Path):
    """Qualquer falha de download/importacao e fatal; nao ha fallback aleatorio."""
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous_hub = torch.hub.get_dir()
    filename = Path(urlsplit(WEIGHTS.url).path).name
    cached = cache_dir / "checkpoints" / filename
    prefix = Path(filename).stem.rsplit("-", 1)[1]
    try:
        torch.hub.set_dir(str(cache_dir))
        # Torchvision verifica o hash no download; verificar tambem cache preexistente.
        if cached.is_file() and not file_sha256(cached).startswith(prefix):
            raise CNNError(f"hash dos pesos ImageNet invalido: {cached}")
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        digest = file_sha256(cached)
        if not digest.startswith(prefix):
            raise CNNError(f"hash dos pesos ImageNet invalido: {cached}")
    except Exception as exc:
        raise CNNError(f"nao foi possivel carregar IMAGENET1K_V1; "
                       f"nenhum fallback aleatorio: {exc}") from exc
    finally:
        torch.hub.set_dir(previous_hub)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model, {
        "enum": "ResNet18_Weights.IMAGENET1K_V1",
        "dataset": "ImageNet-1K",
        "url": WEIGHTS.url,
        "cache_path": str(cached),
        "sha256": digest,
    }


def set_training_phase(model: nn.Module, mode: str) -> dict[str, int]:
    if mode not in MODES:
        raise CNNError(f"modo desconhecido: {mode}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fc.") or
                                 (mode == "layer4+fc" and name.startswith("layer4.")))
        parameter.grad = None
    # Blocos congelados permanecem em eval; inclusive suas BatchNorm.
    model.eval()
    model.fc.train()
    if mode == "layer4+fc":
        model.layer4.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def make_optimizer(model: nn.Module, lr_fc: float, lr_layer4: float,
                   weight_decay: float):
    groups = [{"params": [p for p in model.fc.parameters() if p.requires_grad],
               "lr": lr_fc, "name": "fc"}]
    layer4 = [p for p in model.layer4.parameters() if p.requires_grad]
    if layer4:
        groups.append({"params": layer4, "lr": lr_layer4, "name": "layer4"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def restore_model(checkpoint: dict):
    if checkpoint.get("format_version") != 1:
        raise CNNError("formato de checkpoint nao suportado")
    # Arquitetura vazia somente para restaurar integralmente o estado salvo.
    # Nao e inicializacao aleatoria para treinamento nem fallback ImageNet.
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for name, value in model.state_dict().items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise CNNError(f"checkpoint contem valores nao finitos: {name}")
    model.eval()
    return model
