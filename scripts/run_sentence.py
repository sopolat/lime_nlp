#!/usr/bin/env python3
"""
LIME + HuggingFace BERT demo using gmihaila/bert-base-cased-hatexplain.

- Loads a HF sequence classification model (auto-detects labels from id2label)
- Defines a predict_proba(texts) for LIME (batched, GPU if available)
- Generates a LIME explanation (text printout + optional HTML)

Run:
  python scripts/run_sentence.py \
    --text "These people are awful and should leave." \
    --model gmihaila/bert-base-cased-hatexplain \
    --num-samples 2 \
    --html explanation.html
"""

import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import warnings


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Class names from config (ordered by id)
    id2label = getattr(model.config, "id2label", None)
    if id2label and isinstance(id2label, dict) and len(id2label) == model.config.num_labels:
        class_names = [id2label[i] for i in range(model.config.num_labels)]
    else:
        class_names = [f"LABEL_{i}" for i in range(model.config.num_labels)]

    # Detect problem type for probabilities
    problem_type = getattr(model.config, "problem_type", None)

    @torch.no_grad()
    def predict_proba(texts, batch_size: int = 32, max_length: int = 256):
        """Return an (N, num_labels) numpy array of probabilities."""
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits

            # Probabilities
            if problem_type == "multi_label_classification":
                probs = torch.sigmoid(logits)  # independent per class
                # Normalize rows so LIME gets a distribution (helps stability)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                probs = torch.softmax(logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    return tokenizer, model, class_names, predict_proba


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
    explainer = LimeTextExplainer(class_names=class_names, random_state=42)
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
        html = exp.as_html()
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nSaved HTML explanation to: {os.path.abspath(args.html)}")


if __name__ == "__main__":
    main()
