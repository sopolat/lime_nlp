#!/usr/bin/env python3
"""
LIME + HuggingFace BERT demo using gmihaila/bert-base-cased-hatexplain.

- Loads a HF sequence classification model (auto-detects labels from id2label)
- Defines a predict_proba(texts) for LIME (batched, GPU if available)
- Generates a LIME explanation (text printout + optional HTML)

Run:
  python lime_llm/script_sentence.py \
    --text "can not do politics offline when degenerate scum like you control the offline that why we have to siege kike shill" \
    --model gmihaila/bert-base-cased-hatexplain \
    --num-samples 6 \
    --html original_lime_explanation.html


ToDo:
    Add option to manually provide the samples.

"""

import argparse
import os
import warnings

import numpy as np

from lime.lime_text import LimeTextExplainer
from lime.utils.custom_utils import load_model

SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"




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
