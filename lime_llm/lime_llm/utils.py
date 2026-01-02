#!/usr/bin/env python3
"""
Utils
"""
import json
import os
import re
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import auc, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from lime.lime_text import IndexedString, IndexedCharacters
from .prompts import SYSTEM_PROMPT_VERSIONS, USER_PROMPT_VERSIONS, DATASET_DESCRIPTION
from .args import BaseArgs
from .constants import METHODS


from lime.lime_text import LimeTextExplainer
from lime.utils.custom_utils import load_model

import logging

# This creates a child logger that inherits settings from main
LOG = logging.getLogger(__name__)
warnings.filterwarnings("ignore")



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


def get_prompts(dataset: str, text: str, vocab: list, predicted_label: str, n_per_strategy: int, total_samples: int, args: BaseArgs) -> tuple:
    """Generate dataset-specific prompts with all required parameters."""
    system = SYSTEM_PROMPT_VERSIONS[args.SYSTEM_PROMPT_USE_VERSION]
    user = USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["user_prompt"].format(
        text=text,
        predicted_label=predicted_label,
        vocab=list(vocab),
        vocab_count=len(list(vocab)),
        n_per_strategy=n_per_strategy,
        total_samples=total_samples,
        dataset_description=DATASET_DESCRIPTION[args.DATASET_DESCRIPTION_VERSION][dataset],
    )
    return system, user


# ============================================================================
# LLM INTERFACE
# ============================================================================

def get_llm_client(args: BaseArgs):
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


def call_llm(system_msg: str, user_msg: str, client, args: BaseArgs) -> str:
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


def log_llm_call(idx: str, dataset: str, text: str, predicted_label: str, vocab: str, response: str, args: BaseArgs):
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

def parse_llm_response(response: str, args: BaseArgs) -> dict:
    """Parse LLM JSON response into LIME-compatible format."""
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip())
    data = json.loads(clean)
    if data.get("status") != "OK":
        raise ValueError(f"LLM error: {data.get('error', 'Unknown')}\nLLM OUTPUT: {response}")

    n_required_samples = len(USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["key_outputs"])*args.PERTURBATION_TYPE_SAMPLES
    n_found_samples = len(data.get("samples", []))
    if n_found_samples != n_required_samples:
        LOG.warning(f"⚠️Found {n_found_samples} of samples - required {n_required_samples}")
    texts, masks = [], []
    for sample in data.get("samples", []):
        if sample["strategy"] in USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["key_outputs"]:
            texts.append(sample["text"])
            masks.append(sample["mask"])
        else:
            LOG.warning(f"⚠️Found invalid strategy: {sample["strategy"]} "
                        f"- it should be {str(USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["key_outputs"])}")

    return {"text": texts, "mask": masks}


def get_lime_llm_scores(text: str, dataset: str, model_path: str, llm_client, idx: str, args: BaseArgs) -> np.ndarray:
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
    explainer = LimeTextExplainer(random_state=args.SEED)
    indexed_string = (
        IndexedCharacters(raw_string=text, bow=explainer.bow, mask_string=explainer.mask_string)
        if explainer.char_level else
        IndexedString(raw_string=text, bow=explainer.bow, split_expression=explainer.split_expression,
                      mask_string=explainer.mask_string))
    vocab = [str(val) for val in indexed_string.inverse_vocab]

    # Get dataset-specific prompts
    total_samples = (len(USER_PROMPT_VERSIONS[args.USER_PROMPT_USE_VERSION]["key_outputs"])
                     * args.PERTURBATION_TYPE_SAMPLES)
    system_msg, user_msg = get_prompts(dataset=dataset,
                                       text=text,
                                       vocab=vocab,
                                       predicted_label=predicted_label,
                                       n_per_strategy=args.PERTURBATION_TYPE_SAMPLES,
                                       total_samples=total_samples,
                                       args=args)

    # Get LLM perturbations
    response = call_llm(system_msg, user_msg, llm_client, args=args)

    # Log LLM call
    log_llm_call(idx, dataset, text, predicted_label, vocab, response, args=args)

    try:
        llm_data = parse_llm_response(response, args=args)
    except Exception as e:
        LOG.warning(f"  ✗ LLM parsing failed: {e}")
        return None

    # Run LIME with LLM samples
    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=args.SEED,
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


def plot_curves(df: pd.DataFrame, dataset: str, args: BaseArgs):
    """Plot ROC and PR curves for a dataset."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    df_dataset = df[df["dataset_name"] == dataset].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method, color in zip(METHODS, colors):
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


def aggregate_metrics(df_results: pd.DataFrame, out_json_path: str):
    out = {
        "generated_at": datetime.now().isoformat(),
        "methods": METHODS,
        "datasets": {}
    }

    LOG.info("\n" + "=" * 80)
    LOG.info("AGGREGATE METRICS")
    LOG.info("=" * 80)

    for dataset, dfg in df_results.groupby("dataset_name", dropna=False):
        dataset_key = str(dataset)
        out["datasets"][dataset_key] = {}

        LOG.info(f"\n{dataset_key}:")

        for method in METHODS:
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

    LOG.info("\n" + "=" * 80)
    LOG.info("✓ PIPELINE COMPLETE")

    if out_json_path is None:
        LOG.warning("No outputs saved!")
    else:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            LOG.info(f"Saved: {out_json_path}")

    LOG.info("=" * 80 + "\n")

    return out

def aggregate_metrics_multiseed(df_results: pd.DataFrame, out_json_path: str):
    """
    Computes aggregated metrics (AUC-ROC, AUC-PR) for multi-seed experiments.

    Logic:
    1. Scans columns for exact matches (e.g. 'LIME-LLM', 'Partition SHAP').
    2. Scans columns for seed variations (e.g. 'LIME_42', 'LIME_0') based on METHODS list.
    3. If multiple columns are found for a method, calculates Mean ± Std Dev.
    4. If single column is found, calculates just the Mean (Std Dev = 0).
    """

    # Define your method base names here so the code knows what to look for
    METHODS_TO_ANALYZE = ["LIME", "Partition SHAP", "Integrated Gradient", "LIME-LLM"]

    out = {
        "generated_at": datetime.now().isoformat(),
        "methods": METHODS_TO_ANALYZE,
        "datasets": {}
    }

    LOG.info("\n" + "=" * 80)
    LOG.info(f"AGGREGATE METRICS (Multi-Seed Analysis)")
    LOG.info("=" * 80)

    for dataset, dfg in df_results.groupby("dataset_name", dropna=False):
        dataset_key = str(dataset)
        out["datasets"][dataset_key] = {}

        LOG.info(f"\n{dataset_key}:")

        for method in METHODS_TO_ANALYZE:
            # ---------------------------------------------------------
            # 1. Identify all columns belonging to this method
            # ---------------------------------------------------------
            # Matches:
            #   1. Exact match: "LIME" or "LIME-LLM"
            #   2. Prefix match: "LIME_42" (but NOT "LIME-LLM" when looking for "LIME")
            relevant_cols = [
                c for c in dfg.columns
                if c == method or (c.startswith(f"{method}_"))
            ]

            # CRITICAL: Filter out false positives.
            # If we are looking for "LIME", we do NOT want "LIME-LLM" included.
            if method == "LIME":
                relevant_cols = [c for c in relevant_cols if "LIME-LLM" not in c]

            if not relevant_cols:
                continue

            # ---------------------------------------------------------
            # 2. Calculate Metrics for EACH column (seed) found
            # ---------------------------------------------------------
            seed_results = []

            for col_name in relevant_cols:
                # Subset data for this specific column/seed
                tmp = dfg[[col_name, "words_rationale"]].copy()

                # Convert to numeric and drop NaNs
                tmp[col_name] = pd.to_numeric(tmp[col_name], errors="coerce")
                tmp["words_rationale"] = pd.to_numeric(tmp["words_rationale"], errors="coerce")
                tmp = tmp.dropna(subset=[col_name, "words_rationale"])

                if tmp.empty:
                    continue

                y_true = tmp["words_rationale"].to_numpy(dtype=float)
                y_scores = tmp[col_name].to_numpy(dtype=float)

                # Skip if ground truth has only 1 class (AUC impossible)
                if np.unique(y_true).size < 2:
                    continue

                # Compute standard metrics
                # Assumes compute_metrics returns {'roc_auc': float, 'pr_auc': float}
                m = compute_metrics(y_true, y_scores)
                seed_results.append(m)

            if not seed_results:
                continue

            # ---------------------------------------------------------
            # 3. Aggregate Statistics (Mean ± Std)
            # ---------------------------------------------------------
            n_seeds = len(seed_results)

            # Pivot results: [{'roc': 0.8}, {'roc': 0.82}] -> {'roc': [0.8, 0.82]}
            metrics_lists = {k: [d[k] for d in seed_results] for k in seed_results[0]}

            final_stats = {
                "n_samples": int(len(tmp)),  # Count of last processed column
                "n_seeds": n_seeds,  # Number of columns averaged
            }

            # Build log string dynamically
            log_str = f"  {method:20s}:"

            for metric_name, values in metrics_lists.items():
                mean_val = float(np.mean(values))
                std_val = float(np.std(values)) if n_seeds > 1 else 0.0

                final_stats[metric_name] = mean_val
                final_stats[f"{metric_name}_std"] = std_val

                # Format log output
                if n_seeds > 1:
                    log_str += f" {metric_name.upper()}={mean_val:.4f}±{std_val:.4f}"
                else:
                    log_str += f" {metric_name.upper()}={mean_val:.4f}"

            out["datasets"][dataset_key][method] = final_stats
            LOG.info(log_str + f" (seeds={n_seeds})")

    LOG.info("\n" + "=" * 80)
    LOG.info("✓ PIPELINE COMPLETE")

    if out_json_path is None:
        LOG.warning("No outputs saved!")
    else:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
            LOG.info(f"Saved: {out_json_path}")

    LOG.info("=" * 80 + "\n")

    return out


def plot_curves_multiseed(df: pd.DataFrame, dataset: str, args):
    """
    Plot ROC and PR curves.
    Handles multi-seed methods (like LIME) by plotting Mean line + Shaded Std Dev.
    Handles single-seed methods (like SHAP) by plotting a single line.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Create dataset-specific subset
    df_dataset = df[df["dataset_name"] == dataset].copy()

    # Prepare interpolation grids for averaging curves
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method, color in zip(METHODS, colors):

        # -------------------------------------------------------------
        # 1. Identify Columns (Single vs Multi-Seed)
        # -------------------------------------------------------------
        # Find columns like "LIME_42", "LIME_0" OR exact match "LIME-LLM"
        relevant_cols = [
            c for c in df_dataset.columns
            if c == method or (c.startswith(f"{method}_"))
        ]

        # Strict filter: If method is "LIME", ignore "LIME-LLM"
        if method == "LIME":
            relevant_cols = [c for c in relevant_cols if "LIME-LLM" not in c]

        if not relevant_cols:
            continue

        # -------------------------------------------------------------
        # 2. Collect Curves for all Seeds
        # -------------------------------------------------------------
        tprs = []
        precisions = []
        auc_rocs = []
        auc_prs = []

        for col in relevant_cols:
            # Drop NaNs for this specific column pair
            valid_data = df_dataset[[col, "words_rationale"]].dropna()

            if valid_data.empty:
                continue

            y_true = valid_data["words_rationale"].to_numpy(dtype=float)
            y_scores = valid_data[col].to_numpy(dtype=float)

            if len(np.unique(y_true)) < 2:
                continue

            # --- ROC Calc ---
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # Interpolate TPR to common FPR grid for averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            auc_rocs.append(roc_auc)

            # --- PR Calc ---
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)

            # Interpolate Precision to common Recall grid
            # Note: PR curve returns recall in descending order, so we flip for interp
            interp_prec = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions.append(interp_prec)
            auc_prs.append(pr_auc)

        if not tprs:
            continue

        # -------------------------------------------------------------
        # 3. Aggregate & Plot
        # -------------------------------------------------------------
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0  # Ensure curve ends at 1
        mean_auc_roc = np.mean(auc_rocs)

        mean_precision = np.mean(precisions, axis=0)
        mean_auc_pr = np.mean(auc_prs)

        # -- Calculate Std Dev (only if we have multiple seeds) --
        is_multi_seed = len(tprs) > 1
        std_roc = np.std(tprs, axis=0) if is_multi_seed else None
        std_pr = np.std(precisions, axis=0) if is_multi_seed else None

        # Label formatting
        if is_multi_seed:
            label_roc = f"{method} (AUC={mean_auc_roc:.3f} $\pm${np.std(auc_rocs):.3f})"
            label_pr = f"{method} (AUC={mean_auc_pr:.3f} $\pm${np.std(auc_prs):.3f})"
        else:
            label_roc = f"{method} (AUC={mean_auc_roc:.3f})"
            label_pr = f"{method} (AUC={mean_auc_pr:.3f})"

        # --- Plot ROC ---
        ax1.plot(mean_fpr, mean_tpr, color=color, lw=2, label=label_roc)
        if is_multi_seed:
            ax1.fill_between(mean_fpr,
                             np.maximum(mean_tpr - std_roc, 0),
                             np.minimum(mean_tpr + std_roc, 1),
                             color=color, alpha=0.2)

        # --- Plot PR ---
        ax2.plot(mean_recall, mean_precision, color=color, lw=2, label=label_pr)
        if is_multi_seed:
            ax2.fill_between(mean_recall,
                             np.maximum(mean_precision - std_pr, 0),
                             np.minimum(mean_precision + std_pr, 1),
                             color=color, alpha=0.2)

    # -------------------------------------------------------------
    # 4. Final Formatting
    # -------------------------------------------------------------
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
    ax2.legend(loc="best")  # usually lower left or best
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{args.OUTPUT_DIR}/curves_{dataset}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    LOG.info(f"  ✓ Saved curves: {save_path}")
