#!/usr/bin/env python3
"""Batch ASR evaluation entry point.

Usage:
    python evaluate/batch_test.py
    python evaluate/batch_test.py --config evaluate/config.example.json

默认配置保留了旧脚本的使用习惯：
- 一个任务对应一种语言或一种数据集切片
- 一个 GPU 同时跑一个任务
- 每个模型会和每个任务做一次组合评测
"""

from __future__ import annotations

import argparse
from pathlib import Path

from asr_eval.config import (
    BatchEvalConfig,
    DatasetSpec,
    EvalTaskSpec,
    ModelSpec,
    dump_batch_config,
    load_batch_config,
)


def build_default_config() -> BatchEvalConfig:
    """Provide a runnable template close to the original batch_test.py."""

    return BatchEvalConfig(
        devices=[0, 1, 2, 3, 4, 5, 6, 7],
        output_root="evaluate/results",
        hf_endpoint="https://hf-mirror.com",
        generate_plots=True,
        models=[
            ModelSpec(
                name="model_latin_ok",
                model_path="/data0/test/AI-9-ASR/models/model_latin_ok/whisper-large-v3-turbo-finetune",
            ),
        ],
        tasks=[
            EvalTaskSpec(
                name="latin-test",
                whisper_language="latin",
                dataset=DatasetSpec(
                    source_type="json",
                    data_files="/data0/resources/dataset/test-latin100.json",
                    split="train",
                    audio_column="audio",
                    text_column="sentence",
                    path_from_audio=True,
                ),
            ),
            # 下面是多语种任务示例。需要时取消注释，或者直接改成 --config 方式。
            #
            # EvalTaskSpec(
            #     name="zh-CN",
            #     whisper_language="zh",
            #     dataset=DatasetSpec(
            #         source_type="huggingface",
            #         dataset_name="google/fleurs",
            #         dataset_config="cmn_hans_cn",
            #         split="test[:100]",
            #         cache_dir="/data0/hf-datasets",
            #         audio_column="audio",
            #         text_column="transcription",
            #     ),
            # ),
            # EvalTaskSpec(
            #     name="en-US",
            #     whisper_language="en",
            #     dataset=DatasetSpec(
            #         source_type="huggingface",
            #         dataset_name="google/fleurs",
            #         dataset_config="en_us",
            #         split="test[:100]",
            #         cache_dir="/data0/hf-datasets",
            #         audio_column="audio",
            #         text_column="transcription",
            #     ),
            # ),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper batch evaluation jobs.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON config path. Omit to use the commented default template in this file.",
    )
    parser.add_argument(
        "--dump-default-config",
        default=None,
        help="Write the default config to a JSON file and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = build_default_config()

    if args.dump_default_config:
        dump_batch_config(default_config, args.dump_default_config)
        print(f"default config written to {Path(args.dump_default_config).resolve()}")
        return

    # 只在真正开始评测时才导入运行模块，避免导出配置时也触发 torch 初始化。
    from asr_eval.runner import run_batch_evaluation

    config = load_batch_config(args.config) if args.config else default_config
    session_dir = run_batch_evaluation(config)
    print(f"evaluation finished: {session_dir}")


if __name__ == "__main__":
    main()
