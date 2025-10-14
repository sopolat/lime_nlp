import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
