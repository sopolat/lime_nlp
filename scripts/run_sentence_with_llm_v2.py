#!/usr/bin/env python3
"""
LLM-Enhanced LIME Evaluation Pipeline
Compares LIME-LLM against baseline XAI methods

Version 2 (v2) - Address limitations from v1.
"""

import re
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import IndexedString, IndexedCharacters
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from lime.lime_text import LimeTextExplainer
from lime.utils.custom_utils import load_model


METHODS = ["Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]

LLM_SET = {
    "sonnet45": {
        "model": "claude-sonnet-4-5-20250929",
        "provider": "anthropic"  # "openai" or "anthropic"
    },
    "gpt41": {
        "model": "gpt-4.1-2025-04-14",
        "provider": "openai",
    }
    # "": {
    #     "model": "",
    #     "provider": "",
    # }
}
# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment Settings
TEST_MODE = False  # Set to True to run 1 example per dataset for testing

# LLM Settings
LLM_NAME = "sonnet45"  # sonnet45 gpt41
LLM_PROVIDER = LLM_SET[LLM_NAME]["provider"]
LLM_MODEL = LLM_SET[LLM_NAME]["model"]
SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
TEMPERATURE = 0.0

# Paths
DATA_PATH = "data/xai_combined_df_150_examples.csv"
OUTPUT_DIR = f"outputs/{'test' if TEST_MODE else 'run'}_{LLM_NAME}_{os.path.basename(DATA_PATH).removesuffix(".csv")}"
LOG_FILE = f"{OUTPUT_DIR}/llm_calls.jsonl"

# Model Paths
MODELS = {
    "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
    "hatexplain": "gmihaila/bert-base-cased-hatexplain",
    "cola": "textattack/distilbert-base-uncased-CoLA"
}

# ============================================================================
# DATASET-SPECIFIC PROMPTS
# ============================================================================

# V1 prompt. V2 is worse and deleted.
SYSTEM_PROMPT = """You are an NLP/XAI expert assisting a LIME explainer. You MUST generate EXACTLY {n_samples} mask samples - no more, no fewer. Analyze why the black-box made its prediction, then generate strategic masks over the vocabulary that test your hypotheses. For each mask, create two perturbations: neutral_infill (supports prediction) and boundary_infill (challenges prediction). Output JSON only."""


# Older
# SST2_TRIANGULATION_PROMPT = """
# You are an XAI expert explaining a Sentiment Classifier (SST2).
# TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.
#
# INPUT CONTEXT:
# - Text: "{text}"
# - Text Length: {text_length} words
# - Predicted Label: "{predicted_label}" (Positive/Negative)
# - Vocab: {vocab}
#
# INSTRUCTIONS:
# 1. Generate {n_samples} binary masks representing hypotheses about sentiment drivers.
# 2. For EACH mask, generate 3 distinct samples (The Triangulation):
#
#    Type A: "literal_ablation" (The Drop -> Neutral)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Connect the words into a grammatical sentence using ONLY neutral function words. Keep length close to {text_length} if possible, but prioritize grammar.
#    - CRITICAL: Do NOT add adjectives or adverbs to "fix" the sentiment. If the sentiment word is masked, the result MUST be neutral.
#
#    Type B: "adversarial_flip" (The Floor -> Opposite Label)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Add new context that actively CONTRADICTS the original label.
#    - GOAL: If Original is Positive, write a Negative sentence containing the mask words.
#
#    Type C: "resonant_support" (The Ceiling -> High Confidence)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Write a strong, clear sentence that SUPPORTS the "{predicted_label}".
#    - GOAL: Prove the preserved words are sufficient to maintain the sentiment.
#
# OUTPUT FORMAT (JSON ONLY):
# {{
#   "status": "OK",
#   "sample_count": {n_samples},
#   "samples": [
#     {{
#       "mask": [1,0,1,...],
#       "literal_ablation": {{"text": "complete natural English sentence..."}},
#       "adversarial_flip": {{"text": "complete natural English sentence..."}}
#       "resonant_support": {{"text": "complete natural English sentence..."}}
#     }}
#   ]
# }}
# OR {{"status":"FAIL"}}
#
# REMEMBER: You must provide all {n_samples} samples. Generate COMPLETE, NATURAL sentences. Respect grammaticality constraints."""

# sst2 improvement v1
# SST2_TRIANGULATION_PROMPT = """
# You are an XAI expert explaining a Sentiment Classifier (SST2).
# TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.
#
# INPUT CONTEXT:
# - Text: "{text}"
# - Text Length: {text_length} words
# - Predicted Label: "{predicted_label}" (Positive/Negative)
# - Vocab: {vocab}
#
# INSTRUCTIONS:
# 1. Generate {n_samples} binary masks representing hypotheses about sentiment drivers.
# 2. For EACH mask, generate 3 distinct samples (The Triangulation):
#
#    Type A: "literal_ablation" (The Drop -> Neutral)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Connect the words into a grammatical sentence using ONLY neutral function words (the, a, is).
#    - CRITICAL: Do NOT add adjectives/adverbs to "fix" the sentiment. If the sentiment word is masked, the result MUST be neutral.
#
#    Type B: "adversarial_flip" (The Floor -> Opposite Label)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Add new context that OVERPOWERS the preserved words to flip the label.
#    - STRATEGY: Use sarcasm ("Oh great, another 'terrible' movie") or strong adversaries ("The acting was terrible, but the ending was a MASTERPIECE").
#
#    Type C: "resonant_support" (The Ceiling -> High Confidence)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: maximize support for "{predicted_label}" using ONLY the preserved words.
#    - STRICT PROHIBITION: If the sentiment words are masked, DO NOT add synonyms (e.g., do not swap 'bad' for 'terrible'). Let the support fail if the words are missing.
#
# OUTPUT FORMAT (JSON ONLY):
# {{
#   "status": "OK",
#   "sample_count": {n_samples},
#   "samples": [
#     {{
#       "mask": [1,0,1,...],
#       "literal_ablation": {{"text": "..."}},
#       "adversarial_flip": {{"text": "..."}},
#       "resonant_support": {{"text": "..."}}
#     }}
#   ]
# }}
# """

# sst2 improvement v2
# SST2_TRIANGULATION_PROMPT = """
# You are an XAI expert explaining a Sentiment Classifier (SST2).
# TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.
#
# INPUT CONTEXT:
# - Text: "{text}"
# - Text Length: {text_length} words
# - Predicted Label: "{predicted_label}" (Positive/Negative)
# - Vocab: {vocab}
#
# INSTRUCTIONS:
# 1. Generate {n_samples} binary masks representing hypotheses about sentiment drivers.
# 2. For EACH mask, generate 3 distinct samples (The Triangulation):
#
#    Type A: "literal_ablation" (The Drop -> Neutral)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Connect the words using ONLY neutral function words (the, a, is).
#    - CRITICAL: Do NOT add adjectives/adverbs to "fix" the sentiment. If the sentiment word is masked, the result MUST be neutral.
#
#    Type B: "ambiguous_dilution" (The Gradient -> Low Confidence)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Place the words in a context that makes the sentiment WEAK, MIXED, or UNCLEAR.
#    - STRATEGY: Use questioning ("Was it terrible?"), hearsay ("They said it was terrible"), or mixed signals ("The effects were terrible, but the lighting was okay").
#    - GOAL: Do NOT flip the label completely. Just make the model unsure (reduce confidence).
#
#    Type C: "resonant_support" (The Ceiling -> High Confidence)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Maximize support for "{predicted_label}" using ONLY the preserved words.
#    - STRICT PROHIBITION: If the sentiment words are masked, DO NOT add synonyms.
#
# OUTPUT FORMAT (JSON ONLY):
# {{
#   "status": "OK",
#   "sample_count": {n_samples},
#   "samples": [
#     {{
#       "mask": [1,0,1,...],
#       "literal_ablation": {{"text": "..."}},
#       "ambiguous_dilution": {{"text": "..."}},
#       "resonant_support": {{"text": "..."}}
#     }}
#   ]
# }}
# """

# sst2 improvement v3 - best results
SST2_TRIANGULATION_PROMPT = """
You are an XAI expert explaining a Sentiment Classifier (SST2).
TARGET: Generate a "Consistency Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Positive/Negative)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks. Focus on ISOLATING individual sentiment words (adjectives, adverbs).
2. For EACH mask, generate 3 distinct samples.

   CRITICAL GOAL: All 3 samples must be SEMANTICALLY CONSISTENT with the mask words.
   - NEVER try to flip the label, add irony, or dilute the sentiment.
   - If the mask contains "terrible", the text MUST be negative.

   Type A: "literal_minimalist"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Create the shortest possible grammatical sentence using only neutral function words (the, it, is).
   - PURPOSE: Establish the "pure" signal of the isolated words.

   Type B: "descriptive_framing"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Frame the words in a declarative sentence (e.g., "The review described the scene as [word].").
   - STRICT PROHIBITION: Do NOT add new sentiment words. Rely ONLY on the mask words to drive the prediction.

   Type C: "amplified_context"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Write a fluent sentence where the context supports the mask words naturally.
   - CHECK: Does this text have the exact same sentiment as the isolated words? If yes, good.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_minimalist": {{"text": "..."}},
      "descriptive_framing": {{"text": "..."}},
      "amplified_context": {{"text": "..."}}
    }}
  ]
}}
"""


# COLA_TRIANGULATION_PROMPT = """
# You are an XAI expert explaining a Grammar Classifier (CoLA).
# TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.
#
# INPUT CONTEXT:
# - Text: "{text}"
# - Text Length: {text_length} words
# - Predicted Label: "{predicted_label}" (Acceptable/Unacceptable)
# - Vocab: {vocab}
#
# INSTRUCTIONS:
# 1. Generate {n_samples} binary masks representing hypotheses about grammatical structure.
# 2. For EACH mask, generate 3 distinct samples:
#
#    Type A: "literal_ablation" (The Drop -> Structural Skeleton)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Connect the words using ONLY minimal function words. Do NOT fix agreement if the mask prevents it.
#    - GOAL: Reveal if the preserved words alone force a specific grammatical state.
#
#    Type B: "adversarial_flip" (The Floor -> Opposite Label)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: If Original was Acceptable -> Introduce a specific error. If Unacceptable -> FIX the error.
#
#    Type C: "resonant_support" (The Ceiling -> Same Label)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Rewrite a sentence that maintains the "{predicted_label}" status (grammatical or ungrammatical) using the mask words.
#
# OUTPUT FORMAT (JSON ONLY):
# {{
#   "status": "OK",
#   "sample_count": {n_samples},
#   "samples": [
#     {{
#       "mask": [1,0,1,...],
#       "literal_ablation": {{"text": "complete natural English sentence..."}},
#       "adversarial_flip": {{"text": "complete natural English sentence..."}}
#       "resonant_support": {{"text": "complete natural English sentence..."}}
#     }}
#   ]
# }}
# OR {{"status":"FAIL"}}
#
# REMEMBER: You must provide all {n_samples} samples. Generate COMPLETE, NATURAL sentences. Respect grammaticality constraints."""

COLA_TRIANGULATION_PROMPT = """
You are an XAI expert explaining a Grammar Classifier (CoLA).
TARGET: Generate a "Consistency Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Acceptable/Unacceptable)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks. Focus on isolating structural elements (verbs, agreement, order).
2. For EACH mask, generate 3 distinct samples.

   CRITICAL GOAL: Maintain the GRAMMATICAL STATUS of the mask words.
   - If the mask contains the error (making it Unacceptable), the generated text MUST be Unacceptable.
   - If the mask removes the error, the text MUST be Acceptable.
   - DO NOT FIX GRAMMAR if the error is in the mask.

   Type A: "literal_skeleton"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect the words with minimal function words. Do NOT conjugate or fix agreement.
   - PURPOSE: Isolate the raw structural flaw (or validity).

   Type B: "descriptive_framing"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Embed the words in a quote or simple frame (e.g., He said "[words]").
   - STRICT PROHIBITION: Do NOT correct the grammar inside the frame.

   Type C: "amplified_context"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Write a full sentence using the words.
   - CHECK: If the original text was Unacceptable because of these words, ensure this new sentence is ALSO Unacceptable for the same reason.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_skeleton": {{"text": "..."}},
      "descriptive_framing": {{"text": "..."}},
      "amplified_context": {{"text": "..."}}
    }}
  ]
}}
"""

# HATEXPLAIN_TRIANGULATION_PROMPT = """
# You are an XAI expert explaining a Hate Speech Classifier (HateXplain).
# TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.
#
# INPUT CONTEXT:
# - Text: "{text}"
# - Text Length: {text_length} words
# - Predicted Label: "{predicted_label}" (Hatespeech/Offensive/Normal)
# - Vocab: {vocab}
#
# INSTRUCTIONS:
# 1. Generate {n_samples} binary masks representing hypotheses about toxicity drivers.
# 2. For EACH mask, generate 3 distinct samples:
#
#    Type A: "literal_ablation" (The Drop -> Sanitized/Normal)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Connect words using ONLY neutral function words.
#    - CRITICAL: If a slur is masked, the result MUST be benign.
#
#    Type B: "adversarial_flip" (The Floor -> Counter-Speech/Love)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Actively invert the sentiment to "Love" or "Support" while keeping mask words (e.g., educational context).
#
#    Type C: "resonant_support" (The Ceiling -> Toxic)
#    - RULE: Use ALL Mask=1 words. Exclude Mask=0.
#    - CONSTRAINT: Generate a high-confidence sample matching the "{predicted_label}".
#
# OUTPUT FORMAT (JSON ONLY):
# {{
#   "status": "OK",
#   "sample_count": {n_samples},
#   "samples": [
#     {{
#       "mask": [1,0,1,...],
#       "literal_ablation": {{"text": "complete natural English sentence..."}},
#       "adversarial_flip": {{"text": "complete natural English sentence..."}}
#       "resonant_support": {{"text": "complete natural English sentence..."}}
#     }}
#   ]
# }}
# OR {{"status":"FAIL"}}
#
# REMEMBER: You must provide all {n_samples} samples. Generate COMPLETE, NATURAL sentences. Respect grammaticality constraints."""

HATEXPLAIN_TRIANGULATION_PROMPT = """
You are an XAI expert explaining a Hate Speech Classifier (HateXplain).
TARGET: Generate a "Consistency Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Hatespeech/Offensive/Normal)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks. Focus on isolating specific toxic terms, slurs, and target groups.
2. For EACH mask, generate 3 distinct samples.

   CRITICAL GOAL: Maintain the TOXICITY of the mask words.
   - If the mask contains a slur/hate-term, the text MUST be Hatespeech/Offensive.
   - NEVER use "educational context" or "reclaimed usage" to make a slur sound Normal.
   - If the mask removes the hate-terms, the text MUST be Normal.

   Type A: "literal_minimalist"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect words using ONLY neutral function words.
   - PURPOSE: Expose the raw toxicity of the terms without softening context.

   Type B: "descriptive_framing"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Frame the words simply (e.g., "The message contained [words]").
   - STRICT PROHIBITION: Do NOT dilute the hate speech (e.g., do NOT say "He wrongly said [slur]"). Just present the words.

   Type C: "amplified_context"
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Write a fluent sentence that retains the original hostile intent of the mask words.
   - CHECK: Is this sentence clearly the same label as the original?

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_minimalist": {{"text": "..."}},
      "descriptive_framing": {{"text": "..."}},
      "amplified_context": {{"text": "..."}}
    }}
  ]
}}
"""

PROMPT_TEMPLATES = {
    "sst2": SST2_TRIANGULATION_PROMPT,
    "cola": COLA_TRIANGULATION_PROMPT,
    "hatexplain": HATEXPLAIN_TRIANGULATION_PROMPT
}


def get_prompts(dataset: str, text: str, vocab: list, predicted_label: str, n_samples: int = 10) -> tuple:
    """Generate dataset-specific prompts with all required parameters."""
    system = SYSTEM_PROMPT.format(n_samples=n_samples)
    user = PROMPT_TEMPLATES[dataset].format(
        text=text,
        text_length=len(text.split()),
        predicted_label=predicted_label,
        vocab=list(vocab),
        n_samples=n_samples  # 10 is optimal trade-off based on LLiMe paper
    )
    return system, user


# ============================================================================
# LLM INTERFACE
# ============================================================================

def get_llm_client():
    """Initialize LLM client."""
    if LLM_PROVIDER == "openai":
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
        if LLM_PROVIDER == "openai":
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content
        elif LLM_PROVIDER == "anthropic":
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=4000,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}],
                temperature=TEMPERATURE
            )
            return response.content[0].text
        else:
            raise ValueError(f"INVALID LLM PROVIDER: {LLM_PROVIDER}")
    except Exception as e:
        return json.dumps({"status": "ERROR", "error": str(e)})


def log_llm_call(idx: str, dataset: str, text: str, predicted_label: str, vocab: str, response: str):
    """Log LLM call to JSONL."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(
            json.dumps(
                obj={
                    "idx": idx,
                    "dataset": dataset,
                    "text": text,
                    "predicted_label": predicted_label,
                    "provider": LLM_PROVIDER,
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
        for sample_type_name in ["literal_ablation",
                                 "adversarial_flip",
                                 "resonant_support",
                                 "ambiguous_dilution",  # Testing for sst2
                                 "literal_minimalist",  # Testing for sst2
                                 "descriptive_framing",  # Testing for sst2
                                 "amplified_context",  # Testing for sst2
                                 ]:
            if sample_type_name in sample:
                texts.append(sample[sample_type_name]["text"])
                masks.append(mask)

        # if "literal_ablation" in sample:
        #     texts.append(sample["literal_ablation"]["text"])
        #     masks.append(mask)
        # if "adversarial_flip" in sample:
        #     texts.append(sample["adversarial_flip"]["text"])
        #     masks.append(mask)
        # if "resonant_support" in sample:
        #     texts.append(sample["resonant_support"]["text"])
        #     masks.append(mask)
        # if "ambiguous_dilution" in sample:  # Testing for sst2
        #     texts.append(sample["ambiguous_dilution"]["text"])
        #     masks.append(mask)
        # if "literal_minimalist" in sample:  # Testing for sst2
        #     texts.append(sample["literal_minimalist"]["text"])
        #     masks.append(mask)
        # if "descriptive_framing" in sample:  # Testing for sst2
        #     texts.append(sample["descriptive_framing"]["text"])
        #     masks.append(mask)
        # if "amplified_context" in sample:  # Testing for sst2
        #     texts.append(sample["amplified_context"]["text"])
        #     masks.append(mask)

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
        IndexedString(raw_string=text, bow=explainer.bow, split_expression=explainer.split_expression, mask_string=explainer.mask_string))
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
        print(f"  ✗ LLM parsing failed: {e}")
        return None

    # Run LIME with LLM samples
    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42,
        sentence_transformer_model_name_or_path=SENTENCE_TRANSFORMER_MODEL
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
    plt.savefig(f"{OUTPUT_DIR}/curves_{dataset}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved curves: curves_{dataset}.png")


def aggregate_metrics(df_results: pd.DataFrame, out_json_path: str = "aggregate_metrics.json"):
    out = {
        "generated_at": datetime.now().isoformat(),
        "methods": METHODS,
        "datasets": {}
    }

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    for dataset, dfg in df_results.groupby("dataset_name", dropna=False):
        dataset_key = str(dataset)
        out["datasets"][dataset_key] = {}

        print(f"\n{dataset_key}:")

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

            print(f"  {method:20s}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f} (n={len(tmp)})")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print(f"Saved: {out_json_path}")
    print("=" * 80 + "\n")

    return None

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Setup
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for dataset in MODELS.keys():
        Path(f"{OUTPUT_DIR}/html/{dataset}").mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"LLM-Enhanced LIME Evaluation Pipeline")
    print(f"Mode: {'TEST (1 example per dataset)' if TEST_MODE else 'FULL'}")
    print(f"Provider: {LLM_PROVIDER.upper()}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Test mode: keep all rows for first unique idx per dataset
    if TEST_MODE:
        print("RUNNING LLM 💸💸")
        # Get first unique idx for each dataset
        first_idx_per_dataset = df.groupby("dataset_name")["idx"].first().to_dict()
        # Filter to keep all rows matching these idx values
        df = df[df.apply(lambda row: row["idx"] == first_idx_per_dataset[row["dataset_name"]], axis=1)].reset_index(
            drop=True)
    else:
        if input(f"You are about to run {DATA_PATH} through LLM 💸💸. "
                 f"Do you want to continue? (y/n): ").strip().lower() != 'y':
            print("Exiting...")
            sys.exit(0)


    # Get unique example identifiers
    example_groups = df.groupby(["dataset_name", "idx"], sort=False)

    print(f"Processing {len(example_groups)} examples")
    print(f"Total rows: {len(df)}")
    print(f"Datasets: {df.groupby('dataset_name')['idx'].nunique().to_dict()}\n")

    # Initialize LLM
    llm_client = get_llm_client()

    # Process each example
    results = []

    for example_idx, (group_key, example_data) in enumerate(example_groups):
        dataset, idx = group_key

        # Debug SST2
        # if dataset != "sst2":
        #     continue

        # Get text from first row (all rows have same text for an example)
        text = " ".join(example_data["words"])

        print(f"[{example_idx + 1}/{len(example_groups)}] {dataset} - {idx}")
        print(f"  Text: {text[:80]}...")
        print(f"  Words: {len(example_data)} tokens")

        # Generate LIME-LLM scores
        try:
            scores = get_lime_llm_scores(text, dataset, MODELS[dataset], llm_client, idx)
            if scores is not None:
                # Store LIME-LLM scores back to the example rows
                if len(scores) == len(example_data):
                    example_data_copy = example_data.copy()
                    example_data_copy["LIME-LLM"] = scores
                    results.append(example_data_copy)
                    print(f"  ✓ LIME-LLM scores generated ({len(scores)} tokens)")
                else:
                    print(f"  ✗ Score length mismatch: {len(scores)} vs {len(example_data)} tokens")
                    example_data_copy = example_data.copy()
                    example_data_copy["LIME-LLM"] = None
                    results.append(example_data_copy)
            else:
                print(f"  ✗ Failed to generate scores")
                example_data_copy = example_data.copy()
                example_data_copy["LIME-LLM"] = None
                results.append(example_data_copy)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            example_data_copy = example_data.copy()
            example_data_copy["LIME-LLM"] = None
            results.append(example_data_copy)

    # Combine all results
    df_results = pd.concat(results, ignore_index=True)

    # Save results
    output_cols = ["dataset_name", "idx", "words", "words_rationale",
                   "Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]
    df_results[output_cols].to_csv(f"{OUTPUT_DIR}/results.csv", index=False)
    print(f"\n✓ Results saved: results.csv")

    # Generate evaluation plots for each dataset
    print(f"\nGenerating evaluation plots...")
    for dataset in df_results["dataset_name"].unique():
        print(f"  Processing {dataset}...")
        plot_curves(df_results, dataset)

    # Compute aggregate metrics
    aggregate_metrics(df_results=df_results, out_json_path=f"{OUTPUT_DIR}/aggregate_metrics.json")
    # print(f"\n{'=' * 80}")
    # print("AGGREGATE METRICS")
    # print(f"{'=' * 80}")
    #
    # for dataset in df_results["dataset_name"].unique():
    #     df_dataset = df_results[df_results["dataset_name"] == dataset]
    #     print(f"\n{dataset}:")
    #
    #     for method in ["Partition SHAP", "LIME", "Integrated Gradient", "LIME-LLM"]:
    #         if method not in df_dataset.columns:
    #             continue
    #
    #         # Filter out rows with missing data
    #         valid_rows = df_dataset.dropna(subset=[method, "words_rationale"])
    #
    #         if len(valid_rows) == 0:
    #             continue
    #
    #         y_true = valid_rows["words_rationale"].values
    #         y_scores = valid_rows[method].values
    #
    #         # Ensure both are numeric arrays
    #         try:
    #             y_true = np.array([float(x) for x in y_true])
    #             y_scores = np.array([float(x) for x in y_scores])
    #
    #             if len(np.unique(y_true)) > 1:
    #                 metrics = compute_metrics(y_true, y_scores)
    #                 print(f"  {method:20s}: ROC-AUC={metrics['roc_auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
    #         except (ValueError, TypeError):
    #             print(f"  {method:20s}: Data format error")
    #
    # print(f"\n{'=' * 80}")
    # print(f"✓ PIPELINE COMPLETE")
    # print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()