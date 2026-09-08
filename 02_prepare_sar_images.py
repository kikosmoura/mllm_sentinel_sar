#!/usr/bin/env python3
"""Gera representacoes PNG de chips SAR Sentinel-1.

Para cada chip, a banda 1 e interpretada como VV e a banda 2 como VH. Cada
canal e normalizado de forma independente usando os percentis configurados
(2 e 98 por padrao), clipping e conversao para uint8. A regra e local ao chip:
nenhuma estatistica de outros chips ou de qualquer split e utilizada.

Representacoes geradas:
  * vv:         RGB = (norm(VV), norm(VV), norm(VV))
  * vh:         RGB = (norm(VH), norm(VH), norm(VH))
  * pseudo_rgb: RGB = (norm(VV), norm(VH), norm(VV - VH))

NoData, NaN e infinitos nao participam dos percentis e recebem valor zero no
PNG. Um canal sem nenhum pixel valido ou cujos dois percentis coincidam e
codificado integralmente como zero, evitando uma divisao por zero de maneira
deterministica; canais sem pixels validos tambem geram um aviso no terminal.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    import rasterio
    from PIL import Image
except ImportError as exc:  # pragma: no cover - depende do ambiente
    raise SystemExit(
        "Dependencia ausente. Instale 'numpy', 'rasterio' e 'Pillow' "
        f"antes de executar este script ({exc})."
    ) from exc


REQUIRED_COLUMNS = {"image_id", "s1_path"}
REPRESENTATIONS = ("vv", "vh", "pseudo_rgb")


class InvalidImageError(ValueError):
    """Erro restrito a uma imagem do CSV."""


@dataclass(frozen=True)
class ImageError:
    image_id: str
    s1_path: str
    message: str


@dataclass(frozen=True)
class ImageWarning:
    image_id: str
    s1_path: str
    message: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Converte as bandas VV/VH dos GeoTIFFs Sentinel-1 em PNGs RGB "
            "de 8 bits com normalizacao robusta local por chip."
        ),
        epilog=(
            "Transformacoes: VV=(norm(VV),norm(VV),norm(VV)); "
            "VH=(norm(VH),norm(VH),norm(VH)); pseudo-RGB=(norm(VV),"
            "norm(VH),norm(VV-VH)). NoData/NaN/inf sao ignorados nos "
            "percentis e gravados como zero."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("sen1floods11_classification.csv"),
        help="CSV produzido pelo pipeline de classificacao.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/images"),
        help="Diretorio raiz para vv/, vh/ e pseudo_rgb/.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=2.0,
        help="Percentil inferior usado no clipping de cada canal.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=98.0,
        help="Percentil superior usado no clipping de cada canal.",
    )
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        parser.error(f"--csv nao e um arquivo acessivel: {args.csv}")
    if not (
        np.isfinite(args.lower_percentile)
        and np.isfinite(args.upper_percentile)
        and 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0
    ):
        parser.error(
            "os percentis devem ser finitos e satisfazer "
            "0 <= inferior < superior <= 100"
        )
    return args


def load_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("CSV sem cabecalho")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "CSV sem coluna(s) obrigatoria(s): " + ", ".join(sorted(missing))
            )
        return list(reader), list(reader.fieldnames)


def safe_image_id(raw_image_id: str) -> str:
    image_id = raw_image_id.strip()
    if not image_id or image_id in {".", ".."}:
        raise InvalidImageError("image_id vazio ou invalido")
    if Path(image_id).name != image_id or "/" in image_id or "\\" in image_id:
        raise InvalidImageError(f"image_id contem separador de caminho: {image_id!r}")
    return image_id


def resolve_source_path(raw_path: str, csv_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = csv_path.resolve().parent / path
    return path.resolve()


def normalized_uint8(
    values: np.ndarray,
    valid: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    """Normaliza um canal usando somente seus pixels validos e finitos."""
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = np.zeros(values.shape, dtype=np.uint8)
    if not bool(np.any(valid)):
        return result
    valid_values = np.asarray(values[valid], dtype=np.float64)
    low, high = np.percentile(
        valid_values, [lower_percentile, upper_percentile]
    )
    if not np.isfinite(low) or not np.isfinite(high):
        raise InvalidImageError("percentis do canal nao sao finitos")
    if high <= low:
        return result

    scaled = (valid_values - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    result[valid] = np.rint(scaled * 255.0).astype(np.uint8)
    return result


def validate_rgb_array(name: str, array: np.ndarray, shape: tuple[int, int]) -> None:
    expected_shape = (shape[0], shape[1], 3)
    if array.shape != expected_shape:
        raise InvalidImageError(
            f"representacao {name} tem forma {array.shape}; esperado {expected_shape}"
        )
    if array.dtype != np.uint8:
        raise InvalidImageError(
            f"representacao {name} tem dtype {array.dtype}; esperado uint8"
        )
    if not bool(np.all(np.isfinite(array))):
        raise InvalidImageError(f"representacao {name} contem NaN ou infinito")
    if int(array.min()) < 0 or int(array.max()) > 255:
        raise InvalidImageError(f"representacao {name} fora do intervalo [0, 255]")


def prepare_representations(
    s1_path: Path,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[dict[str, np.ndarray], tuple[int, int], list[str]]:
    try:
        with rasterio.open(s1_path) as dataset:
            if dataset.count < 2:
                raise InvalidImageError(
                    "GeoTIFF S1 possui menos de duas bandas (VV e VH): "
                    f"{dataset.count}"
                )
            bands = dataset.read([1, 2], masked=True)
            shape = (dataset.height, dataset.width)
    except InvalidImageError:
        raise
    except Exception as exc:
        raise InvalidImageError(f"nao foi possivel ler o GeoTIFF: {exc}") from exc

    vv_masked, vh_masked = bands[0], bands[1]
    vv = np.asarray(vv_masked.data, dtype=np.float32)
    vh = np.asarray(vh_masked.data, dtype=np.float32)
    vv_valid = ~np.ma.getmaskarray(vv_masked) & np.isfinite(vv)
    vh_valid = ~np.ma.getmaskarray(vh_masked) & np.isfinite(vh)
    common_valid = vv_valid & vh_valid
    warnings: list[str] = []
    if not bool(np.any(vv_valid)):
        warnings.append("VV sem pixels validos; canal codificado como zero")
    if not bool(np.any(vh_valid)):
        warnings.append("VH sem pixels validos; canal codificado como zero")
    if not bool(np.any(common_valid)):
        warnings.append(
            "VV e VH sem pixels validos em comum; VV-VH codificado como zero"
        )

    difference = np.zeros(shape, dtype=np.float32)
    np.subtract(vv, vh, out=difference, where=common_valid)

    vv_8bit = normalized_uint8(
        vv, vv_valid, lower_percentile, upper_percentile
    )
    vh_8bit = normalized_uint8(
        vh, vh_valid, lower_percentile, upper_percentile
    )
    difference_8bit = normalized_uint8(
        difference, common_valid, lower_percentile, upper_percentile
    )

    representations = {
        "vv": np.repeat(vv_8bit[:, :, np.newaxis], 3, axis=2),
        "vh": np.repeat(vh_8bit[:, :, np.newaxis], 3, axis=2),
        "pseudo_rgb": np.stack(
            (vv_8bit, vh_8bit, difference_8bit), axis=2
        ),
    }
    for name, array in representations.items():
        validate_rgb_array(name, array, shape)
    return representations, shape, warnings


def write_and_validate_pngs(
    image_id: str,
    representations: dict[str, np.ndarray],
    output_dirs: dict[str, Path],
    shape: tuple[int, int],
) -> dict[str, Path]:
    final_paths = {
        name: output_dirs[name] / f"{image_id}.png" for name in REPRESENTATIONS
    }
    temporary_paths = {
        name: path.with_name(f".{path.name}.tmp")
        for name, path in final_paths.items()
    }
    try:
        for name in REPRESENTATIONS:
            Image.fromarray(representations[name]).save(
                temporary_paths[name], format="PNG", compress_level=9
            )
        for name, path in temporary_paths.items():
            with Image.open(path) as image:
                image.load()
                if image.mode != "RGB":
                    raise InvalidImageError(
                        f"PNG {name} salvo no modo {image.mode}; esperado RGB"
                    )
                if image.size != (shape[1], shape[0]):
                    raise InvalidImageError(
                        f"PNG {name} salvo com dimensoes {image.size}; "
                        f"esperado {(shape[1], shape[0])}"
                    )
        for name in REPRESENTATIONS:
            temporary_paths[name].replace(final_paths[name])
    except Exception:
        for path in temporary_paths.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return final_paths


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, _ = load_rows(args.csv)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"[ERRO FATAL] Nao foi possivel ler o CSV: {exc}", file=sys.stderr)
        return 1

    output_root = args.output_dir.resolve()
    output_dirs = {name: output_root / name for name in REPRESENTATIONS}
    try:
        for directory in output_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[ERRO FATAL] Nao foi possivel criar a saida: {exc}", file=sys.stderr)
        return 1

    id_counts: dict[str, int] = {}
    for row in rows:
        key = row.get("image_id", "").strip().casefold()
        id_counts[key] = id_counts.get(key, 0) + 1
    duplicate_ids = {key for key, count in id_counts.items() if key and count > 1}

    errors: list[ImageError] = []
    warnings: list[ImageWarning] = []
    successful_ids: set[str] = set()
    expected_shape: tuple[int, int] | None = None
    for row in rows:
        raw_image_id = row.get("image_id", "")
        raw_s1_path = row.get("s1_path", "")
        try:
            image_id = safe_image_id(raw_image_id)
            if image_id.casefold() in duplicate_ids:
                raise InvalidImageError(f"image_id duplicado no CSV: {image_id!r}")
            s1_path = resolve_source_path(raw_s1_path, args.csv)
            if not s1_path.is_file():
                raise InvalidImageError(f"arquivo S1 inexistente: {s1_path}")
            representations, shape, image_warnings = prepare_representations(
                s1_path,
                args.lower_percentile,
                args.upper_percentile,
            )
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise InvalidImageError(
                    f"dimensoes {shape} diferem do padrao {expected_shape}"
                )
            write_and_validate_pngs(
                image_id, representations, output_dirs, shape
            )
            successful_ids.add(image_id.casefold())
            for message in image_warnings:
                warning = ImageWarning(image_id, str(s1_path), message)
                warnings.append(warning)
                print(
                    f"[AVISO] Imagem {image_id!r}: {message}; S1={s1_path}",
                    file=sys.stderr,
                )
        except Exception as exc:
            error = ImageError(raw_image_id, raw_s1_path, str(exc))
            errors.append(error)
            print(
                f"[ERRO] Imagem {raw_image_id!r}: {exc}; S1={raw_s1_path}",
                file=sys.stderr,
            )

    # Conte usando os IDs preservados, inclusive em sistemas case-sensitive.
    png_counts = {
        name: sum(
            1
            for row in rows
            if row.get("image_id", "").strip().casefold() in successful_ids
            and (directory / f"{row.get('image_id', '').strip()}.png").is_file()
        )
        for name, directory in output_dirs.items()
    }

    print("\nRelatorio de preparacao SAR")
    print(f"Registros no CSV: {len(rows)}")
    print(f"Imagens processadas: {len(successful_ids)}")
    print(f"PNGs VV: {png_counts['vv']}")
    print(f"PNGs VH: {png_counts['vh']}")
    print(f"PNGs pseudo-RGB: {png_counts['pseudo_rgb']}")
    print(f"Dimensoes de saida: {expected_shape or 'N/A'}")
    print(
        "Percentis por canal/chip: "
        f"{args.lower_percentile:g}-{args.upper_percentile:g}"
    )
    print(f"Avisos encontrados: {len(warnings)}")
    print(f"Erros encontrados: {len(errors)}")
    print(f"Diretorio de saida: {output_root}")

    counts_are_consistent = all(
        count == len(successful_ids) for count in png_counts.values()
    )
    if errors or not counts_are_consistent:
        if not counts_are_consistent:
            print(
                "[ERRO] A quantidade de PNGs nao coincide com a de imagens "
                "processadas.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
