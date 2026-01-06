#!/usr/bin/env python3
"""
LLM-Enhanced LIME Evaluation Pipeline
Compares LIME-LLM against baseline XAI methods

"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import wandb

from lime_llm.utils import get_llm_client, get_lime_llm_scores, plot_curves_multiseed, wandb_log_curves, aggregate_metrics_multiseed

from dotenv import load_dotenv
from lime_llm.log import get_logger
from lime_llm.args import BaseArgs
from lime_llm.constants import TASK_MODELS, METHODS, LLM_SET

warnings.filterwarnings("ignore")


class Args(BaseArgs):
    """
    LLM-Enhanced LIME Evaluation Pipeline arguments.
    """

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    SEED = 42  # SEEDS = [42, 0, 1, 123, 1234, 2023, 2024, 7, 10, 99]
    TEST_MODE = False  # Set to True to run 1 example per dataset for testing
    USE_CASH = True # Set to True to use already made llm calls
    SYSTEM_PROMPT_USE_VERSION = "v9"
    USER_PROMPT_USE_VERSION = "v9"
    DATASET_DESCRIPTION_VERSION = "v9"
    LLM_NAME = "gemini3"  # sonnet45 gpt5 gpt41
    SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
    TEMPERATURE = 1  # Open AI needs to be 1 and Anthropic is 0.0
    DATA_SAMPLES = 150  # 30 60 150
    PERTURBATION_TYPE_SAMPLES = 10  # 30 TOTAL samples is ideal
    WANDB_ENABLED = False
    WANDB_PROJECT = "lime-nlp"
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # optional
    WANDB_MODE = os.getenv("WANDB_MODE", "online")  # "online" | "offline" | "disabled"

    LLM_PROVIDER = LLM_SET[LLM_NAME]["provider"]
    LLM_MODEL = LLM_SET[LLM_NAME]["model"]
    # DATA_PATH = f"data/original/xai_combined_df_{DATA_SAMPLES}_examples.csv"
    DATA_PATH = f"data/test/xai_combined_df_test_human_rationale_{DATA_SAMPLES}_examples.csv"  # TEST
    OUTPUT_DIR = f"outputs/{'test' if TEST_MODE else 'run'}_{LLM_NAME}_{os.path.basename(DATA_PATH).removesuffix(".csv")}_sys{SYSTEM_PROMPT_USE_VERSION}_usr{USER_PROMPT_USE_VERSION}_datadesc{DATASET_DESCRIPTION_VERSION}"
    LOG_FILE = f"{OUTPUT_DIR}/llm_calls.jsonl"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

args = Args()
LOG = get_logger(f"{args.OUTPUT_DIR}/pipeline.log")
load_dotenv()


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

    # Load data
    df_all_data = pd.read_csv('data/updated_baselines_test/xai_combined_df_150_examples_all_explanations_lime_seeds.csv')  # Test
    df_all_data["idx"] = df_all_data["idx"].astype(str)
    df_all_data["dataset_name"] = df_all_data["dataset_name"].astype(str)
    df_all_data["words"] = df_all_data["words"].astype(str)
    df_all_data["words_rationale"] = df_all_data["words_rationale"].astype(int)
    # Load Sample data
    df_sample = pd.read_csv(args.DATA_PATH)
    idx_sample = df_sample["idx"].astype(str).tolist()
    LOG.info(f"df_sample: {len(idx_sample)}")
    df = df_all_data[df_all_data["idx"].isin(idx_sample)].reset_index(drop=True)

    # Test mode: keep all rows for first unique idx per dataset
    if args.TEST_MODE:
        LOG.info("RUNNING LLM 💸💸")
        # Get first unique idx for each dataset
        first_idx_per_dataset = df.groupby("dataset_name")["idx"].first().to_dict()
        # Filter to keep all rows matching these idx values
        df = df[df.apply(lambda row: row["idx"] == first_idx_per_dataset[row["dataset_name"]], axis=1)].reset_index(
            drop=True)
    else:
        if input(f"You are about to run {args.DATA_PATH} through LLM 💸💸. "
                 f"Do you want to continue? (y/n): ").strip().lower() != 'y':
            LOG.info("Exiting...")
            sys.exit(0)

    # --- wandb init (minimal) ---
    run = None
    if args.WANDB_ENABLED and wandb is not None and args.WANDB_MODE != "disabled":
        run = wandb.init(
            project=args.WANDB_PROJECT,
            entity=args.WANDB_ENTITY,
            name=os.path.basename(args.OUTPUT_DIR),
            config=args.to_dict(),
            tags=[args.LLM_NAME, args.LLM_PROVIDER, f"sys{args.SYSTEM_PROMPT_USE_VERSION}",
                  f"usr{args.USER_PROMPT_USE_VERSION}"],
            mode=args.WANDB_MODE,
        )

    if run:
        wandb.log({
            "data/rows": int(len(df)),
            "data/examples": int(len(df.groupby(["dataset_name", "idx"]))),
            "data/examples_per_dataset": df.groupby("dataset_name")["idx"].nunique().to_dict(),
        })

    for dataset in TASK_MODELS.keys():
        Path(f"{args.OUTPUT_DIR}/html/{dataset}").mkdir(parents=True, exist_ok=True)

    # Get unique example identifiers
    example_groups = df.groupby(["dataset_name", "idx"], sort=False)

    LOG.info(f"Processing {len(example_groups)} examples")
    LOG.info(f"Total rows: {len(df)}")
    LOG.info(f"Datasets: {df.groupby('dataset_name')['idx'].nunique().to_dict()}\n")

    # Initialize LLM
    llm_client = get_llm_client(args=args)

    # Process each example
    results = []

    trouble_examples = []
    for example_idx, (group_key, example_data) in enumerate(example_groups):
        dataset, idx = group_key

        # DEBUG DATASET
        # if dataset != "cola":  # "hatexplain" "cola" "sst2"
        #     continue

        # Get text from first row (all rows have same text for an example)
        text = " ".join(example_data["words"])

        LOG.info(f"[{example_idx + 1}/{len(example_groups)}] {dataset} - {idx}")
        LOG.info(f"  Text: {text[:80]}...")
        LOG.info(f"  Words: {len(example_data)} tokens")

        # Generate LIME-LLM scores
        try:
            scores = get_lime_llm_scores(text, dataset, TASK_MODELS[dataset], llm_client, idx, args=args)
            if scores is not None:
                # Store LIME-LLM scores back to the example rows
                if len(scores) == len(example_data):
                    example_data_copy = example_data.copy()
                    example_data_copy["LIME-LLM"] = scores
                    results.append(example_data_copy)
                    LOG.info(f"  ✓ LIME-LLM scores generated ({len(scores)} tokens)")
                else:
                    LOG.info(f"  ✗ Score length mismatch: {len(scores)} vs {len(example_data)} tokens")
                    example_data_copy = example_data.copy()
                    example_data_copy["LIME-LLM"] = None
                    results.append(example_data_copy)
            else:
                LOG.warning(f"  ✗ Failed to generate scores")
                example_data_copy = example_data.copy()
                example_data_copy["LIME-LLM"] = None
                results.append(example_data_copy)
        except Exception as e:
            LOG.warning(f"  ✗ Error: {e}")
            example_data_copy = example_data.copy()
            example_data_copy["LIME-LLM"] = None
            results.append(example_data_copy)
            trouble_examples.append(f"{idx}|{dataset}|'{text}'")

        if run:
            wandb.log(
                {
                    "progress/example_idx": example_idx + 1,
                    "progress/examples_total": len(example_groups),
                    "progress/failed_examples": len(trouble_examples),
                    "example/ok": 0 if scores is None else 1,
                },
                step=example_idx + 1
            )

    # Combine all results
    df_results = pd.concat(results, ignore_index=True)

    # Save results
    lime_columns = [val for val in df.columns.tolist() if "lime" in val.lower()]
    output_cols = ["dataset_name", "idx", "words", "words_rationale"] + lime_columns + ["Partition SHAP", "Integrated Gradient", "LIME-LLM"]
    df_results[output_cols].to_csv(f"{args.OUTPUT_DIR}/results.csv", index=False)
    LOG.warning(f"\n✓ Results saved: results.csv")
    if run:
        wandb.log({"files/results_csv": wandb.Table(dataframe=df_results[output_cols].head(50))})

    # Generate evaluation plots for each dataset
    LOG.info(f"\nGenerating evaluation plots...")
    for dataset in df_results["dataset_name"].unique():
        LOG.info(f"  Processing {dataset}...")
        plot_curves_multiseed(df_results, dataset, args=args)
        if run:
            img_path = f"{args.OUTPUT_DIR}/curves_{dataset}.png"
            if os.path.exists(img_path):
                wandb.log({f"curves/{dataset}": wandb.Image(img_path)})
            wandb_log_curves(df_results, dataset, run, METHODS)

    # Compute aggregate metrics
    # agg = aggregate_metrics(df_results=df_results, out_json_path=f"{args.OUTPUT_DIR}/aggregate_metrics.json")
    agg = aggregate_metrics_multiseed(df_results=df_results, out_json_path=f"{args.OUTPUT_DIR}/aggregate_metrics.json")


    if run and agg:
        # log scalar metrics
        for dset, methods in agg.get("datasets", {}).items():
            for method, m in methods.items():
                wandb.log({
                    f"roc_auc_{dset}/{method}": m["roc_auc"],
                })
                wandb.log({
                    f"pr_auc_{dset}/{method}": m["pr_auc"],
                })

    LOG.warning(f"Failed examples {len(trouble_examples)}/{len(example_groups)}\n{trouble_examples}")

    if run:
        # trouble examples table
        rows = []
        for s in trouble_examples:
            parts = s.split("|", 2)
            if len(parts) == 3:
                rows.append(parts)
        if rows:
            wandb.log({"trouble_examples": wandb.Table(columns=["idx", "dataset", "text"], data=rows)})

        # artifacts (results + metrics + llm log if present)
        art = wandb.Artifact(name=os.path.basename(args.OUTPUT_DIR), type="run_outputs")
        for p in [
            f"{args.OUTPUT_DIR}/results.csv",
            f"{args.OUTPUT_DIR}/aggregate_metrics.json",
            args.LOG_FILE,
        ]:
            if os.path.exists(p):
                art.add_file(p)

        run.log_artifact(art)

        art = wandb.Artifact(name="prompts", type="code")
        art.add_file("lime_llm/lime_llm/prompts.py")  # logs the actual .py file
        run.log_artifact(art)

        run.finish()


if __name__ == "__main__":
    main()
