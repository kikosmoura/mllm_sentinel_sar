#!/usr/bin/env python3
"""Avalia a mesma MLLM base carregando um adapter PEFT treinado."""

from mllm.evaluation import create_evaluation_parser, run_evaluation, validate_args


def main() -> int:
    parser = create_evaluation_parser(
        description=(
            "Avalia validation e test com o mesmo pipeline zero-shot, "
            "acrescentando um adapter PEFT/LoRA."
        ),
        default_output_dir="results/mllm_finetuned",
        require_adapter=True,
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return run_evaluation(args, run_name="finetuned")


if __name__ == "__main__":
    raise SystemExit(main())
