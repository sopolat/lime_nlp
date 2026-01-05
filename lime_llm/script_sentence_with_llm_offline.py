#!/usr/bin/env python3
"""
Using "script_sentence_with_llm.py output file "llm_calls.jsonl" to re-run evaluation offline
"""
import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from lime_llm.args import BaseArgs
from lime_llm.constants import LLM_SET
from lime_llm.log import get_logger
from lime_llm.utils import aggregate_metrics_multiseed, plot_curves_multiseed

warnings.filterwarnings("ignore")

import pandas as pd


class Args(BaseArgs):
    """
    LLM-Enhanced LIME Evaluation Pipeline arguments.
    """

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    SEED = 0
    TEST_MODE = False  # Set to True to run 1 example per dataset for testing
    SYSTEM_PROMPT_USE_VERSION = "v9"
    USER_PROMPT_USE_VERSION = "v9"
    DATASET_DESCRIPTION_VERSION = "v9"
    LLM_NAME = "gpt5"  # sonnet45 gpt41 gpt5
    DATA_SAMPLES = 150  # 30 60 150
    DATA_PATH = f"data/test/xai_combined_df_test_human_rationale_{DATA_SAMPLES}_examples.csv"
    SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
    TEMPERATURE = 0.0
    PERTURBATION_TYPE_SAMPLES = 10  # 30 TOTAL samples is ideal
    WANDB_ENABLED = False
    WANDB_PROJECT = "lime-nlp"
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # optional
    WANDB_MODE = os.getenv("WANDB_MODE", "online")  # "online" | "offline" | "disabled"

    LLM_PROVIDER = LLM_SET[LLM_NAME]["provider"]
    LLM_MODEL = LLM_SET[LLM_NAME]["model"]
    # DATA_PATH = f"data/original/xai_combined_df_{DATA_SAMPLES}_examples.csv"
    OUTPUT_DIR = f"outputs/{'test' if TEST_MODE else 'run'}_{LLM_NAME}_{os.path.basename(DATA_PATH).removesuffix(".csv")}_sys{SYSTEM_PROMPT_USE_VERSION}_usr{USER_PROMPT_USE_VERSION}_datadesc{DATASET_DESCRIPTION_VERSION}"
    LOG_FILE = f"{OUTPUT_DIR}/llm_calls.jsonl"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


args = Args()
LOG = get_logger(__file__.replace(".py", ".log").replace("/lime_llm/", "/logs/"))
# Load environment variables from .env file
load_dotenv()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def read_jsonl(file_path):
    """Read a JSONL file with multi-line JSON objects."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(content):
        content = content[idx:].lstrip()  # Skip whitespace
        if not content:
            break
        try:
            obj, end_idx = decoder.raw_decode(content)
            data.append(obj)
            idx += end_idx
        except json.JSONDecodeError:
            break

    return data


def main():
    """
    MAIN PIPELINE
    """

    # Setup
    LOG.info(f"\n{'=' * 80}")
    LOG.info(f"LLM-Enhanced LIME Evaluation Pipeline")
    LOG.info(f"Mode: {'TEST (1 example per dataset)' if args.TEST_MODE else 'FULL'}")
    LOG.info(f"Provider: {args.LLM_PROVIDER.upper()}")
    LOG.info(f"Output: {args.OUTPUT_DIR}")
    LOG.info(f"{'=' * 80}\n")

    # Load llm calls LOG_FILE
    llm_calls = pd.read_csv(f"{args.OUTPUT_DIR}/results.csv")
    llm_calls["idx"] = llm_calls["idx"].astype(str)
    llm_calls["dataset_name"] = llm_calls["dataset_name"].astype(str)
    llm_calls["words"] = llm_calls["words"].astype(str)
    llm_calls["words_rationale"] = llm_calls["words_rationale"].astype(int)
    LOG.info(f"llm_calls: {len(llm_calls)}")

    # Load data
    df_all_data = pd.read_csv('data/updated_baselines_test/xai_combined_df_150_examples_all_explanations_lime_seeds.csv')
    df_all_data["idx"] = df_all_data["idx"].astype(str)
    df_all_data["dataset_name"] = df_all_data["dataset_name"].astype(str)
    df_all_data["words"] = df_all_data["words"].astype(str)
    df_all_data["words_rationale"] = df_all_data["words_rationale"].astype(int)
    # Load Sample data
    df_sample = pd.read_csv(args.DATA_PATH)
    idx_sample = df_sample["idx"].astype(str).tolist()
    LOG.info(f"df_sample: {len(idx_sample)}")
    df = df_all_data[df_all_data["idx"].isin(idx_sample)].reset_index(drop=True)

    common_columns = ["dataset_name", "idx", "words", "words_rationale"]
    lime_columns = [val for val in df.columns.tolist() if "lime" in val.lower()]
    data_columns = common_columns + ['Partition SHAP', 'Integrated Gradient'] + lime_columns
    results_columns = common_columns + ['LIME-LLM']

    df_results = df[data_columns].merge(llm_calls[results_columns], on=common_columns,
                                        how="left").dropna().reset_index(drop=True)

    # _ = aggregate_metrics(df_results=updated_results, out_json_path=None)
    _ = aggregate_metrics_multiseed(df_results=df_results, out_json_path=None)

    args.OUTPUT_DIR = 'outputs/test-run'

    # Generate evaluation plots for each dataset
    LOG.info(f"\nGenerating evaluation plots...")
    for dataset in df_results["dataset_name"].unique():
        LOG.info(f"  Processing {dataset}...")
        plot_curves_multiseed(df_results, dataset, args=args)


if __name__ == "__main__":
    main()
