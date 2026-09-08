#!/usr/bin/env python3
"""Avalia uma MLLM original, sem exemplos de treino no contexto."""

from mllm.evaluation import create_evaluation_parser, run_evaluation, validate_args


def main() -> int:
    parser = create_evaluation_parser(
        description="Avalia a MLLM base em zero-shot sobre validation e test.",
        default_output_dir="results/mllm_zero_shot",
        require_adapter=False,
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return run_evaluation(args, run_name="zero_shot")


if __name__ == "__main__":
    raise SystemExit(main())
