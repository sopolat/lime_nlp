#!/usr/bin/env python3
"""
LIME + HuggingFace BERT demo using gmihaila/bert-base-cased-hatexplain.

- Loads a HF sequence classification model (auto-detects labels from id2label)
- Defines a predict_proba(texts) for LIME (batched, GPU if available)
- Generates a LIME explanation (text printout + optional HTML)

Run:
  python scripts/run_sentence.py \
    --text "can not do politics offline when degenerate scum like you control the offline that why we have to siege kike shill" \
    --model gmihaila/bert-base-cased-hatexplain \
    --num-samples 2 \
    --html custom_llm_lime_explanation.html



ToDo:
    Add option to manually provide the samples.

"""

import json
import argparse
import os
import warnings

import numpy as np

from lime.lime_text import LimeTextExplainer
from lime.utils.custom_utils import load_model

SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
GPT_MODEL = "gpt-3.5-turbo"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


def chat_gpt_call(system_message, prompt_message, llm_client, llm_model=GPT_MODEL):
    """Calls the ChatGPT API with a given instruction and prompt."""
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,  # Or another suitable model like "gpt-4"
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def anthropic_call(system_message, prompt_message, llm_client, llm_model=ANTHROPIC_MODEL):
    """Calls the Anthropic API with a given prompt and optional system message."""
    try:
        messages = [{"role": "user", "content": prompt_message}]
        response = llm_client.messages.create(
            model=llm_model,  # Or another suitable model
            max_tokens=1000,
            system=system_message,
            messages=messages
        )

        return response.content[0].text

    except Exception as e:
        return f"An error occurred: {e}"


def restructure_to_llm_sample_data(data):
    """
    Restructure nested JSON to extract tokens, masks, and sample types.

    Args:
        data: Dict containing the JSON response structure

    Returns:
        tuple: (llm_sample_data, masks, sample_types)
            - llm_sample_data: list of lists, each inner list contains tokens
            - masks: list of lists, each inner list contains the mask for that sample
            - sample_types: list of strings indicating infill type ("neutral" or "boundary")

    Raises:
        ValueError: If status is not "OK"
    """
    # Check status and raise error if not OK
    status = data.get("status", "UNKNOWN")
    if status != "OK":
        error_msg = data.get("error", "Unknown error occurred")
        raise ValueError(f"API returned non-OK status: '{status}'. Error: {error_msg}")

    llm_sample_data = []
    masks = []
    sample_types = []

    for sample in data.get("samples", []):
        mask = sample.get("mask", [])

        # Add neutral_infill tokens as a string
        if "neutral_infill" in sample:
            tokens = sample["neutral_infill"].get("tokens", [])
            llm_sample_data.append(" ".join(tokens))  # Join with whitespace
            masks.append(mask)
            sample_types.append("neutral")

        # Add boundary_infill tokens as a string
        if "boundary_infill" in sample:
            tokens = sample["boundary_infill"].get("tokens", [])
            llm_sample_data.append(" ".join(tokens))  # Join with whitespace
            masks.append(mask)
            sample_types.append("boundary")

    return llm_sample_data, masks, sample_types

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="LIME explanation with HF BERT")
    parser.add_argument("--text", default="These people are awful and should leave.",
                        help="Input sentence to explain")
    parser.add_argument("--model", default="gmihaila/bert-base-cased-hatexplain",
                        help="HF model repo or local path")
    parser.add_argument("--num-features", type=int, default=10,
                        help="Number of features to show in the explanation")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of perturbed samples LIME generates (BERT is slow; 1000 is a good start)")
    parser.add_argument("--html", default="", help="Optional path to save HTML explanation")
    args = parser.parse_args()

    tokenizer, model, class_names, predict_proba = load_model(args.model)

    # Get probabilities and predicted class
    probs = predict_proba([args.text])[0]
    pred_idx = int(np.argmax(probs))
    print(f"\nInput: {args.text!r}")
    print("Classes:", class_names)
    pretty_probs = ", ".join(f"{cls}={p:.3f}" for cls, p in zip(class_names, probs))
    print(f"Predicted: {class_names[pred_idx]}  ({pretty_probs})")

    # Get LLM Samples
    llm_response_dict = json.loads("""{
      "status": "OK",
      "samples": [
        {
          "mask": [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1],
          "neutral_infill": {
            "tokens": ["don't", "want", "to", "talk", "to", "anyone", "degenerate", "scum", "like", "you", "when", "you", "post", "because", "now", "they", "will", "get", "siege", "kike", "shill"]
          },
          "boundary_infill": {
            "tokens": ["why", "do", "we", "have", "to", "be", "degenerate", "scum", "like", "you", "on", "the", "web", "just", "to", "make", "a", "point", "siege", "kike", "shill"]
          }
        },
        {
          "mask": [1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
          "neutral_infill": {
            "tokens": ["can", "not", "do", "politics", "offline", "when", "evil", "monsters", "here", "you", "control", "the", "offline", "that", "why", "we", "have", "to", "attack", "these", "people"]
          },
          "boundary_infill": {
            "tokens": ["can", "not", "do", "politics", "offline", "when", "some", "people", "out", "you", "control", "the", "offline", "that", "why", "we", "have", "to", "protest", "this", "thing"]
          }
        },
        {
          "mask": [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
          "neutral_infill": {
            "tokens": ["they", "will", "never", "politics", "offline", "when", "degenerate", "scum", "like", "you", "control", "the", "offline", "that", "why", "we", "have", "to", "siege", "kike", "shill"]
          },
          "boundary_infill": {
            "tokens": ["we", "should", "do", "politics", "offline", "when", "degenerate", "scum", "like", "you", "control", "the", "offline", "that", "why", "we", "have", "to", "siege", "kike", "shill"]
          }
        }
      ]
    }"""
                                    )
    text_data, masks, sample_types = restructure_to_llm_sample_data(data=llm_response_dict)

    llm_sample_data = {"text": text_data,
                       "mask": masks,
                       }

    # LIME explanation
    explainer = LimeTextExplainer(
        class_names=class_names,
        random_state=42,
        sentence_transformer_model_name_or_path=SENTENCE_TRANSFORMER_MODEL,
    )
    exp = explainer.explain_instance(
        args.text,
        predict_proba,
        num_features=args.num_features,
        labels=[pred_idx],
        num_samples=args.num_samples,
        llm_sample_data=llm_sample_data,

    )

    print("\nTop features for predicted class:")
    # for word, weight in exp.as_list(label=pred_idx):
    #     print(f"  {word:>20s} : {weight:+.4f}")

    if args.html:
        path = os.path.join("outputs", args.html)
        html = exp.as_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nSaved HTML explanation to: {path}")


if __name__ == "__main__":
    main()
