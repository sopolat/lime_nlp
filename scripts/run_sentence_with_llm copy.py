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
            llm_sample_data.append(sample["neutral_infill"].get("text"))  # Join with whitespace
            masks.append(mask)
            sample_types.append("neutral")

        # Add boundary_infill tokens as a string
        if "boundary_infill" in sample:
            llm_sample_data.append(sample["boundary_infill"].get("text"))  # Join with whitespace
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
  "sample_count": 10,
  "samples": [
    {
      "mask": [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1],
      "neutral_infill": {"text": "degenerate scum spreading kike shill propaganda"},
      "boundary_infill": {"text": "degenerate scum confronting kike shill peacefully"}
    },
    {
      "mask": [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
      "neutral_infill": {"text": "degenerate people trying to siege kike violently"},
      "boundary_infill": {"text": "degenerate groups discussing siege kike situation"}
    },
    {
      "mask": [1,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
      "neutral_infill": {"text": "can do politics offline you control everything badly"},
      "boundary_infill": {"text": "can do politics offline you control discussions properly"}
    },
    {
      "mask": [0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,0,0],
      "neutral_infill": {"text": "hateful scum like you we must siege immediately"},
      "boundary_infill": {"text": "brave scum like you we could siege safely"}
    },
    {
      "mask": [0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
      "neutral_infill": {"text": "politics offline when control the situation harshly"},
      "boundary_infill": {"text": "politics offline when control the discourse fairly"}
    },
    {
      "mask": [0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,1,1,0,0,0],
      "neutral_infill": {"text": "degenerate scum you why we have to attack"},
      "boundary_infill": {"text": "degenerate scum you why we have to talk"}
    },
    {
      "mask": [1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
      "neutral_infill": {"text": "can not like you control the that rudely"},
      "boundary_infill": {"text": "can not like you control the that nicely"}
    },
    {
      "mask": [0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0],
      "neutral_infill": {"text": "when degenerate that why have to fight"},
      "boundary_infill": {"text": "when degenerate that why have to unite"}
    },
    {
      "mask": [0,1,1,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1,0,0],
      "neutral_infill": {"text": "not do you the we to siege violently"},
      "boundary_infill": {"text": "not do you the we to siege peacefully"}
    },
    {
      "mask": [0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1],
      "neutral_infill": {"text": "politics offline like have kike shill attacking"},
      "boundary_infill": {"text": "politics offline like have kike shill talking"}
    }
  ]
}"""
   )
#
#     llm_response_dict = json.loads("""{
#   "status": "OK",
#   "samples": [
#     {
#       "mask": [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1],
#       "neutral_infill": {"text": "degenerate scum kike shill spreading hate"},
#       "boundary_infill": {"text": "degenerate scum discussing kike shill terminology"}
#     },
#     {
#       "mask": [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
#       "neutral_infill": {"text": "degenerate people trying to siege kike community"},
#       "boundary_infill": {"text": "degenerate protesters siege kike museum discussion"}
#     },
#     {
#       "mask": [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1],
#       "neutral_infill": {"text": "these scum like you shill always causing trouble"},
#       "boundary_infill": {"text": "people scum like you shill in the debate"}
#     },
#     {
#       "mask": [1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0],
#       "neutral_infill": {"text": "can do politics you control siege them all"},
#       "boundary_infill": {"text": "can do politics you control siege the event"}
#     },
#     {
#       "mask": [0,0,0,0,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,0],
#       "neutral_infill": {"text": "offline when degenerate the kike causes problems"},
#       "boundary_infill": {"text": "offline when degenerate the kike history discussed"}
#     },
#     {
#       "mask": [0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,0,0],
#       "neutral_infill": {"text": "degenerate scum you why we need to leave"},
#       "boundary_infill": {"text": "degenerate scum you why we should talk"}
#     },
#     {
#       "mask": [0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,1,1,0,0,0],
#       "neutral_infill": {"text": "politics you the that have to stop now"},
#       "boundary_infill": {"text": "politics you the that have to discuss more"}
#     },
#     {
#       "mask": [0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0],
#       "neutral_infill": {"text": "not scum like control siege the place"},
#       "boundary_infill": {"text": "not scum like control siege peacefully"}
#     },
#     {
#       "mask": [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0],
#       "neutral_infill": {"text": "offline you we have to kike get out"},
#       "boundary_infill": {"text": "offline you we have to kike learn more"}
#     },
#     {
#       "mask": [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1],
#       "neutral_infill": {"text": "degenerate scum like you kike shill forever"},
#       "boundary_infill": {"text": "degenerate scum like you kike shill history"}
#     },
#     {
#       "mask": [0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1],
#       "neutral_infill": {"text": "offline degenerate scum siege kike shill"},
#       "boundary_infill": {"text": "offline degenerate scum like control siege kike shill"}
#     },
#     {
#       "mask": [1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0],
#       "neutral_infill": {"text": "can do politics you have to siege kike shill"},
#       "boundary_infill": {"text": "can do offline you to see kike shill"}
#     },
#     {
#       "mask": [0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
#       "neutral_infill": {"text": "do politics offline"},
#       "boundary_infill": {"text": "not politics offline when scum like you control siege kike shill"}
#     },
#     {
#       "mask": [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
#       "neutral_infill": {"text": "when"},
#       "boundary_infill": {"text": "when you the that why we to siege kike"}
#     },
#     {
#       "mask": [1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0],
#       "neutral_infill": {"text": "can degenerate like control to kike"},
#       "boundary_infill": {"text": "can politics offline when scum you have to shill"}
#     },
#     {
#       "mask": [0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0],
#       "neutral_infill": {"text": "not offline you to kike"},
#       "boundary_infill": {"text": "do scum to siege shill"}
#     },
#     {
#       "mask": [0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
#       "neutral_infill": {"text": "degenerate offline"},
#       "boundary_infill": {"text": "do when scum like you control the why we have siege kike shill"}
#     },
#     {
#       "mask": [0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0],
#       "neutral_infill": {"text": "offline siege"},
#       "boundary_infill": {"text": "offline degenerate siege kike"}
#     },
#     {
#       "mask": [1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0],
#       "neutral_infill": {"text": "can control siege shill"},
#       "boundary_infill": {"text": "can politics you to siege kike shill"}
#     },
#     {
#       "mask": [1,1,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0],
#       "neutral_infill": {"text": "can not offline scum the that we have to kike shill"},
#       "boundary_infill": {"text": "can not offline like control"}
#     }
#   ]
# }"""
#     )

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
    for word, weight in exp.as_list(label=pred_idx):
        print(f"  {word:>20s} : {weight:+.4f}")

    if args.html:
        path = os.path.join("outputs", args.html)
        html = exp.as_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nSaved HTML explanation to: {path}")


if __name__ == "__main__":
    main()
