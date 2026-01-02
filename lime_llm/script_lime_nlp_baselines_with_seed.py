# -*- coding: utf-8 -*-
"""lime_nlp_baselines.ipynb

conda create --name lime_nlp_baselines python=3.12.12 lime shap captum python-dotenv --y

Python Script:
/Users/georgemihaila/miniconda3/envs/lime_nlp_baselines/bin/python lime_llm/script_lime_nlp_baselines_with_seed.py --seed 42

Shell Script:
bash lime_llm/run_script_lime_nlp_baselines_with_seed.sh


"""
import os

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import lime.lime_text
from lime_llm.constants import TASK_MODELS

# Load environment variables from .env file
load_dotenv()

import lime.lime_text
import argparse

from dotenv import load_dotenv

import lime.lime_text

import torch
import numpy as np
import random
import lime.lime_text
import shap
from captum.attr import LayerIntegratedGradients
from lime_llm.log import get_logger

# Load environment variables from .env file
LOG = get_logger(__file__.replace(".py", ".log").replace("/lime_llm/", "/logs/"))
load_dotenv()


def evaluate_xai_aligned(text, model, tokenizer, seed=42):
    """
    Returns LIME, SHAP, and IG scores aligned to text.split().
    Output: (lime_scores, shap_scores, ig_scores) as lists of floats.
    """
    # 1. SEEDING & SETUP
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = model.device

    # 2. HELPER: PREDICT PROBA
    def predict_proba(texts):
        if isinstance(texts, np.ndarray): texts = texts.tolist()
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad(): outputs = model(**inputs)
        return torch.softmax(outputs.logits, dim=1).cpu().numpy()

    # 3. LIME (Bag-of-Words Force)
    lime_exp = lime.lime_text.LimeTextExplainer(
        class_names=model.config.id2label, random_state=seed, verbose=False, bow=True, split_expression=r'\s+'
    ).explain_instance(text, predict_proba, num_features=len(text.split()), num_samples=5000)

    # Extract LIME scores mapped to words
    pred_class = list(lime_exp.local_exp.keys())[0]
    lime_map = {lime_exp.domain_mapper.indexed_string.word(f): s for f, s in lime_exp.local_exp[pred_class]}
    lime_scores = [float(lime_map.get(w, 0.0)) for w in text.split()]

    # only run all of the explanations for seed 42
    if seed != 42:
        return lime_scores, None, None

    # 4. SHAP & IG (Sub-token Level)
    # -- SHAP
    shap_vals = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer), seed=seed)([text])
    shap_raw = shap_vals.values[0]
    if shap_raw.ndim > 1: shap_raw = shap_raw[:, np.argmax(np.abs(shap_raw).mean(0))]  # Select top class

    # -- IG
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(device)
    target_cls = torch.argmax(model(inputs.input_ids).logits).item()

    emb_layer = getattr(model, model.config.model_type).embeddings  # Auto-find embeddings
    lig = LayerIntegratedGradients(lambda x: model(x).logits, emb_layer)
    ig_raw = lig.attribute(inputs.input_ids, target=target_cls, n_steps=50, return_convergence_delta=False)
    ig_raw = ig_raw.sum(dim=2).squeeze().detach().cpu().numpy()

    # 5. AGGREGATION (Sub-tokens -> Words)
    words = text.split()
    shap_out, ig_out = np.zeros(len(words)), np.zeros(len(words))

    # Create character spans for original words: "Hello world" -> [(0,5), (6,11)]
    cursor, word_spans = 0, []
    for w in words:
        word_spans.append((cursor, cursor + len(w)))
        cursor += len(w) + 1  # +1 for whitespace

    # Map tokens to words via offsets
    offsets = inputs['offset_mapping'][0].cpu().numpy()
    for i, (start, end) in enumerate(offsets):
        if start == end: continue  # Skip special tokens ([CLS], [SEP])
        # Find which word span contains this token center
        token_center = start + (end - start) // 2
        for w_idx, (w_start, w_end) in enumerate(word_spans):
            if w_start <= token_center < w_end:
                shap_out[w_idx] += shap_raw[i]
                ig_out[w_idx] += ig_raw[i]
                break

    return lime_scores, shap_out.tolist(), ig_out.tolist()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run XAI baselines with a specific seed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    LOG.info(args)

    """# Load models and tokenizers"""

    # 1. Load a standard pre-trained model (e.g., Sentiment Analysis)
    device = "cpu"

    task_model_tokenizers = {
        ds: {"model": AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval(),
             "tokenizer": AutoTokenizer.from_pretrained(model_name, use_fast=True)}
        for ds, model_name in TASK_MODELS.items()}

    DATA_SAMPLES = 150  # all samples are compounding
    # DATA_PATH = f"data/original/xai_combined_df_{DATA_SAMPLES}_examples.csv"
    # NEW_DATA_PATH = f"data/updated_baselines_{args.seed}seed"

    DATA_PATH = f"data/test/xai_combined_df_test_human_rationale_{DATA_SAMPLES}_examples.csv"  # Test data
    NEW_DATA_PATH = f"data/updated_baselines_test"

    os.makedirs(NEW_DATA_PATH, exist_ok=True)
    if args.seed == 42:
        NEW_DATA_FILE_PATH = os.path.join(NEW_DATA_PATH, f"xai_combined_df_{DATA_SAMPLES}_examples_all_explanations_{args.seed}seed.csv")
    else:
        NEW_DATA_FILE_PATH = os.path.join(NEW_DATA_PATH, f"xai_combined_df_{DATA_SAMPLES}_examples_{args.seed}seed.csv")

    df = pd.read_csv(DATA_PATH)#[:10]

    # 2. Loop through each example
    lime_scores, shap_scores, ig_scores = [], [], []

    example_groups = df.groupby(["dataset_name", "idx"], sort=False)

    for example_idx, (group_key, example_data) in tqdm(enumerate(example_groups),
                                                       desc="baselines",
                                                       total=len(example_groups)):
        dataset, idx = group_key

        # # Get text from first row (all rows have same text for an example)
        text = " ".join(example_data["words"])

        l_scores, s_scores, i_scores = evaluate_xai_aligned(text=text, seed=args.seed, **task_model_tokenizers[dataset])

        lime_scores += l_scores

        if args.seed == 42:
            shap_scores += s_scores
            ig_scores += i_scores

    new_df = df[['dataset_name', 'idx', 'words', 'words_rationale']].copy()
    new_df["LIME"] = lime_scores

    if args.seed == 42:
        new_df["Partition SHAP"] = shap_scores
        new_df["Integrated Gradient"] = ig_scores

    new_df.to_csv(NEW_DATA_FILE_PATH, index=False)
    LOG.warning(f"SAVED '{NEW_DATA_FILE_PATH}'")


if __name__ == "__main__":
    main()
