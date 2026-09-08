"""Prompt unico dos experimentos MLLM zero-shot e fine-tuned."""

CLASSIFICATION_PROMPT = """This image is a visual representation derived from
Sentinel-1 SAR data.

Classify it according to the presence of surface water.

Answer with exactly one label:
WATER
NON_WATER"""
