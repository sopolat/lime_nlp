#!/usr/bin/env python3
"""
LLM-Enhanced LIME Evaluation Pipeline
Compares LIME-LLM against baseline XAI methods

"""

import re
import json
import logging, sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from lime.lime_text import IndexedString, IndexedCharacters
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from lime.lime_text import LimeTextExplainer
from lime.utils.custom_utils import load_model
from prompts import SYSTEM_PROMPT_VERSIONS, USER_PROMPT_VERSIONS, DATASET_DESCRIPTION


class Args(object):
    """
    LLM-Enhanced LIME Evaluation Pipeline arguments.
    """

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    TEST_MODE = False  # Set to True to run 1 example per dataset for testing
    SYSTEM_PROMPT_USE_VERSION = "v9"
    USER_PROMPT_USE_VERSION = "v9"
    DATASET_DESCRIPTION_VERSION = "v9"
    LLM_NAME = "sonnet45"  # sonnet45 gpt41
    SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
    TEMPERATURE = 0.0
    DATA_SAMPLES = 30  # 30 60 150
    WANDB_ENABLED = True
    WANDB_PROJECT = "lime-nlp"
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # optional
    WANDB_MODE = os.getenv("WANDB_MODE", "online")  # "online" | "offline" | "disabled"

    # Fixed
    TASK_MODELS = {
        "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
        "hatexplain": "gmihaila/bert-base-cased-hatexplain",
        "cola": "textattack/distilbert-base-uncased-CoLA"
    }
    METHODS = ["Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]
    LLM_SET = {
        "sonnet45": {
            "model": "claude-sonnet-4-5-20250929",
            "provider": "anthropic"  # "openai" or "anthropic"
        },
        "gpt41": {
            "model": "gpt-4.1-2025-04-14",
            "provider": "openai",
        },
    }
    LLM_PROVIDER = LLM_SET[LLM_NAME]["provider"]
    LLM_MODEL = LLM_SET[LLM_NAME]["model"]
    DATA_PATH = f"data/xai_combined_df_{DATA_SAMPLES}_examples.csv"
    OUTPUT_DIR = f"outputs/{'test' if TEST_MODE else 'run'}_{LLM_NAME}_{os.path.basename(DATA_PATH).removesuffix(".csv")}_sys{SYSTEM_PROMPT_USE_VERSION}_usr{USER_PROMPT_USE_VERSION}_datadesc{DATASET_DESCRIPTION_VERSION}"
    LOG_FILE = f"{OUTPUT_DIR}/llm_calls.jsonl"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """
        Combine class and instance attributes
        """
        dict_arguments = {**self.__class__.__dict__}
        dict_arguments = {k:str(v) for k, v in dict_arguments.items()}
        dict_arguments.update({
            "system_prompt_version": SYSTEM_PROMPT_VERSIONS[args.SYSTEM_PROMPT_USE_VERSION],
            "user_prompt_version": USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION],
            "dataset_description_version": DATASET_DESCRIPTION[args.DATASET_DESCRIPTION_VERSION], })
        return dict_arguments


def get_logger(log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    Logger
    """
    log = logging.getLogger("lime-llm")
    if log.handlers:  # already configured (prevents duplicate logs)
        return log

    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    log.addHandler(sh)
    log.addHandler(fh)
    return log


args = Args()
LOG = get_logger(f"{args.OUTPUT_DIR}/pipeline.log")
# Load environment variables from .env file
load_dotenv()


def wandb_log_curves(df_results: pd.DataFrame, dataset: str, run, methods, n_grid: int = 201):
    """
    Log “all methods” ROC + PR as W&B line-series charts (no images)
    """
    dfd = df_results[df_results["dataset_name"] == dataset]

    fpr_grid = np.linspace(0.0, 1.0, n_grid)
    rec_grid = np.linspace(0.0, 1.0, n_grid)

    keys, ys_roc, ys_pr = [], [], []

    for method in methods:
        if method not in dfd.columns:
            continue

        tmp = dfd[[method, "words_rationale"]].copy()
        tmp[method] = pd.to_numeric(tmp[method], errors="coerce")
        tmp["words_rationale"] = pd.to_numeric(tmp["words_rationale"], errors="coerce")
        tmp = tmp.dropna()
        if tmp.empty:
            continue

        y_true = tmp["words_rationale"].to_numpy(dtype=float)
        y_score = tmp[method].to_numpy(dtype=float)
        if np.unique(y_true).size < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        tpr_i = np.interp(fpr_grid, fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        order = np.argsort(rec)
        prec_i = np.interp(rec_grid, rec[order], prec[order])

        keys.append(method)
        ys_roc.append(tpr_i.tolist())
        ys_pr.append(prec_i.tolist())

    if not keys:
        return None, None

    run.log({
        f"curves/{dataset}/roc": wandb.plot.line_series(
            fpr_grid.tolist(), ys_roc, keys=keys, title=f"ROC - {dataset}", xname="False Positive Rate"
        ),
        f"curves/{dataset}/pr": wandb.plot.line_series(
            rec_grid.tolist(), ys_pr, keys=keys, title=f"PR - {dataset}", xname="Recall"
        ),
    })
    return (fpr_grid, ys_roc, keys), (rec_grid, ys_pr, keys)


def get_prompts(dataset: str, text: str, vocab: list, predicted_label: str, n_samples: int = 10) -> tuple:
    """Generate dataset-specific prompts with all required parameters."""
    system = SYSTEM_PROMPT_VERSIONS[args.SYSTEM_PROMPT_USE_VERSION].format(n_samples=n_samples, )
    user = USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["user_prompt"].format(
        text=text,
        text_length=len(text.split()),
        predicted_label=predicted_label,
        vocab=list(vocab),
        vocab_count=len(list(vocab)),
        n_samples=n_samples,  # 10 is optimal trade-off based on LLiMe paper
        dataset_description=DATASET_DESCRIPTION[args.DATASET_DESCRIPTION_VERSION][dataset],
    )
    return system, user


# ============================================================================
# LLM INTERFACE
# ============================================================================

def get_llm_client():
    """Initialize LLM client."""
    if args.LLM_PROVIDER == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI")
        if not api_key:
            raise ValueError("OPENAI environment variable not set in .env file")
        return OpenAI(api_key=api_key)
    else:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC")
        if not api_key:
            raise ValueError("ANTHROPIC environment variable not set in .env file")
        return Anthropic(api_key=api_key)


def call_llm(system_msg: str, user_msg: str, client) -> str:
    """Call LLM and return response."""
    try:
        if args.LLM_PROVIDER == "openai":
            response = client.chat.completions.create(
                model=args.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=args.TEMPERATURE
            )
            return response.choices[0].message.content
        elif args.LLM_PROVIDER == "anthropic":
            response = client.messages.create(
                model=args.LLM_MODEL,
                max_tokens=4000,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
                temperature=args.TEMPERATURE
            )
            return response.content[0].text
        else:
            raise ValueError(f"INVALID LLM PROVIDER: {args.LLM_PROVIDER}")
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def log_llm_call(idx: str, dataset: str, text: str, predicted_label: str, vocab: str, response: str):
    """Log LLM call to JSONL."""
    with open(args.LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(
            json.dumps(
                obj={
                    "idx": idx,
                    "dataset": dataset,
                    "text": text,
                    "predicted_label": predicted_label,
                    "provider": args.LLM_PROVIDER,
                    "timestamp": datetime.now().isoformat(),
                    "vocab": vocab,
                    "n_vocab": len(vocab),
                    "response": response,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================================
# LIME PROCESSING
# ============================================================================

def parse_llm_response(response: str) -> dict:
    """Parse LLM JSON response into LIME-compatible format."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
    data = json.loads(clean)
    if data.get("status") != "OK":
        raise ValueError(f"LLM error: {data.get('error', 'Unknown')}")

    texts, masks = [], []
    for sample in data.get("samples", []):
        mask = sample.get("mask", [])
        for key_output in USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["key_outputs"]:
            if key_output in sample:
                texts.append(sample[key_output]["text"])  # Needs added.
                masks.append(mask)

    return {"text": texts, "mask": masks}


def get_lime_llm_scores(text: str, dataset: str, model_path: str, llm_client, idx: str) -> np.ndarray:
    """Generate LIME-LLM explanation scores (word-level)."""
    # Load model first to get predicted label
    tokenizer, model, class_names, predict_proba = load_model(model_path)

    # Get prediction
    probs = predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))
    predicted_label = class_names[pred_idx]

    if dataset == "sst2":
        predicted_label = class_names[pred_idx].lower()

    elif dataset == "cola":
        mapping_labels = {"LABEL_0": "unacceptable",
                          "LABEL_1": "acceptable"}
        predicted_label = mapping_labels[predicted_label]

    elif dataset == "hatexplain":
        mapping_labels = {"LABEL_0": "hatespeech",
                          "LABEL_1": "normal",
                          "LABEL_2": "offensive",
                          }
        predicted_label = mapping_labels[predicted_label]

    # Extract vocabulary (unique words from text)
    explainer = LimeTextExplainer(random_state=42)
    indexed_string = (
        IndexedCharacters(raw_string=text, bow=explainer.bow, mask_string=explainer.mask_string)
        if explainer.char_level else
        IndexedString(raw_string=text, bow=explainer.bow, split_expression=explainer.split_expression,
                      mask_string=explainer.mask_string))
    vocab = [str(val) for val in indexed_string.inverse_vocab]

    # Get dataset-specific prompts
    system_msg, user_msg = get_prompts(dataset, text, vocab, predicted_label, n_samples=10)

    # Get LLM perturbations
    response = call_llm(system_msg, user_msg, llm_client)

    # Log LLM call
    log_llm_call(idx, dataset, text, predicted_label, vocab, response)

    try:
        llm_data = parse_llm_response(response)
    except Exception as e:
        LOG.warning(f"  ✗ LLM parsing failed: {e}")
        return None

    # Run LIME with LLM samples
    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42,
        sentence_transformer_model_name_or_path=args.SENTENCE_TRANSFORMER_MODEL
    )

    exp = explainer.explain_instance(
        text,
        predict_proba,
        labels=[pred_idx],
        num_samples=1000,
        llm_sample_data=llm_data
    )

    # Extract feature importance scores (word-level)
    tokens = text.split()
    scores = np.zeros(len(tokens))

    explanation_map = dict(exp.as_list(label=pred_idx))

    for i, token in enumerate(tokens):
        # Match token to explanation (case-insensitive, partial match)
        for word, weight in explanation_map.items():
            if word.lower() in token.lower() or token.lower() in word.lower():
                scores[i] = abs(weight)
                break

    return scores


# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> dict:
    """Compute ROC-AUC and PR-AUC."""
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan}
    return {
        "roc_auc": roc_auc_score(y_true, y_scores),
        "pr_auc": average_precision_score(y_true, y_scores)
    }


def plot_curves(df: pd.DataFrame, dataset: str):
    """Plot ROC and PR curves for a dataset."""
    methods = ["Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    df_dataset = df[df["dataset_name"] == dataset].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method, color in zip(methods, colors):
        if method not in df_dataset.columns:
            continue

        # Filter valid data (word-level)
        valid_data = df_dataset.dropna(subset=[method, "words_rationale"])

        if len(valid_data) == 0:
            continue

        try:
            y_true = np.array([float(x) for x in valid_data["words_rationale"].values])
            y_scores = np.array([float(x) for x in valid_data[method].values])
        except (ValueError, TypeError):
            continue

        if len(np.unique(y_true)) < 2:
            continue

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{method} (AUC={roc_auc:.3f})")

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        ax2.plot(recall, precision, color=color, lw=2, label=f"{method} (AUC={pr_auc:.3f})")

    # ROC formatting
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title(f"ROC Curve - {dataset}", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # PR formatting
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title(f"Precision-Recall Curve - {dataset}", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{args.OUTPUT_DIR}/curves_{dataset}.png", dpi=300, bbox_inches="tight")
    plt.close()
    LOG.info(f"  ✓ Saved curves: curves_{dataset}.png")


def aggregate_metrics(df_results: pd.DataFrame, out_json_path: str = "aggregate_metrics.json"):
    out = {
        "generated_at": datetime.now().isoformat(),
        "methods": args.METHODS,
        "datasets": {}
    }

    LOG.info("\n" + "=" * 80)
    LOG.info("AGGREGATE METRICS")
    LOG.info("=" * 80)

    for dataset, dfg in df_results.groupby("dataset_name", dropna=False):
        dataset_key = str(dataset)
        out["datasets"][dataset_key] = {}

        LOG.info(f"\n{dataset_key}:")

        for method in args.METHODS:
            if method not in dfg.columns:
                continue

            tmp = dfg[[method, "words_rationale"]].copy()
            tmp[method] = pd.to_numeric(tmp[method], errors="coerce")
            tmp["words_rationale"] = pd.to_numeric(tmp["words_rationale"], errors="coerce")
            tmp = tmp.dropna(subset=[method, "words_rationale"])

            if tmp.empty:
                continue

            y_true = tmp["words_rationale"].to_numpy(dtype=float)
            y_scores = tmp[method].to_numpy(dtype=float)

            if np.unique(y_true).size < 2:
                # ROC/PR AUC not meaningful if only one class present
                continue

            m = compute_metrics(y_true, y_scores)  # expects {"roc_auc": ..., "pr_auc": ...}
            out["datasets"][dataset_key][method] = {
                "n": int(len(tmp)),
                "roc_auc": float(m["roc_auc"]),
                "pr_auc": float(m["pr_auc"]),
            }

            LOG.info(f"  {method:20s}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f} (n={len(tmp)})")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    LOG.info("\n" + "=" * 80)
    LOG.info("✓ PIPELINE COMPLETE")
    LOG.info(f"Saved: {out_json_path}")
    LOG.info("=" * 80 + "\n")

    return out


# ============================================================================
# MAIN PIPELINE
# ============================================================================

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
    df = pd.read_csv(args.DATA_PATH)

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

    for dataset in args.TASK_MODELS.keys():
        Path(f"{args.OUTPUT_DIR}/html/{dataset}").mkdir(parents=True, exist_ok=True)

    # Get unique example identifiers
    example_groups = df.groupby(["dataset_name", "idx"], sort=False)

    LOG.info(f"Processing {len(example_groups)} examples")
    LOG.info(f"Total rows: {len(df)}")
    LOG.info(f"Datasets: {df.groupby('dataset_name')['idx'].nunique().to_dict()}\n")

    # Initialize LLM
    llm_client = get_llm_client()

    # Process each example
    results = []

    trouble_examples = []
    for example_idx, (group_key, example_data) in enumerate(example_groups):
        dataset, idx = group_key

        # Debug SST2
        # if dataset != "sst2":
        #     continue

        # Get text from first row (all rows have same text for an example)
        text = " ".join(example_data["words"])

        LOG.info(f"[{example_idx + 1}/{len(example_groups)}] {dataset} - {idx}")
        LOG.info(f"  Text: {text[:80]}...")
        LOG.info(f"  Words: {len(example_data)} tokens")

        # Generate LIME-LLM scores
        try:
            scores = get_lime_llm_scores(text, dataset, args.TASK_MODELS[dataset], llm_client, idx)
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
    output_cols = ["dataset_name", "idx", "words", "words_rationale",
                   "Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]
    df_results[output_cols].to_csv(f"{args.OUTPUT_DIR}/results.csv", index=False)
    LOG.warning(f"\n✓ Results saved: results.csv")
    if run:
        wandb.log({"files/results_csv": wandb.Table(dataframe=df_results[output_cols].head(50))})

    # Generate evaluation plots for each dataset
    LOG.info(f"\nGenerating evaluation plots...")
    for dataset in df_results["dataset_name"].unique():
        LOG.info(f"  Processing {dataset}...")
        plot_curves(df_results, dataset)
        if run:
            img_path = f"{args.OUTPUT_DIR}/curves_{dataset}.png"
            if os.path.exists(img_path):
                wandb.log({f"curves/{dataset}": wandb.Image(img_path)})
            wandb_log_curves(df_results, dataset, run, args.METHODS)

    # Compute aggregate metrics
    # aggregate_metrics(df_results=df_results, out_json_path=f"{args.OUTPUT_DIR}/aggregate_metrics.json")
    agg = aggregate_metrics(df_results=df_results, out_json_path=f"{args.OUTPUT_DIR}/aggregate_metrics.json")

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

        p = Path("prompts.py")
        art = wandb.Artifact(name="prompts", type="code")
        art.add_file(str(p))  # logs the actual .py file
        run.log_artifact(art)

        run.finish()


if __name__ == "__main__":
    main()
