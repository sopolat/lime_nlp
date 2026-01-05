#!/usr/bin/env python3
"""
Implementation of LLIME paper for baseline comparison
"""
import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import openai
from lime_llm.utils import load_model
from lime_llm.constants import TASK_MODELS

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import pairwise_distances
from openai import OpenAI
import re
from dotenv import load_dotenv

load_dotenv()


class LLiMe:
    def __init__(self, classifier_fn, openai_client, model="gpt-4"):
        """
        Implementation of LLiMe (Angiulli et al., 2025).

        Args:
            classifier_fn: Function taking a list of texts and returning probas [N, 2].
            openai_client: Initialized openai.OpenAI() client.
            model: OpenAI model name (paper uses GPT-4).
        """
        self.predict_fn = classifier_fn
        self.client = openai_client
        self.model = model
        # LLiMe uses a binary bag-of-words representation (Section 4.1.2)
        self.vectorizer = CountVectorizer(binary=True)

    def _call_llm(self, prompt, n=1):
        """Helper to call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                n=1
            )
            return response.choices.message.content
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def _get_neighbor_prompt(self, text, target_class):
        """
        Exact prompt from LLiMe Section 4.1.1 for neighborhood generation.
        """
        return f"""You are an advanced, context-aware paraphrasing system with a precise understanding of semantics. Your task is to generate paraphrased sentences for a given input sentence, "S", ensuring alignment with a provided class "C".
### GUIDELINES:
Generate 10 paraphrased sentences that align with the semantics of the provided class.
Each paraphrased sentence must:
- Preserve the structure of the original sentence while adapting to the designated class.
- Retain as much of the original vocabulary as possible unless modifications are necessary for semantic alignment.
- Begin with the "$" symbol for clear separation.
### OUTPUT:
$ [Paraphrased sentence 1]
$ [Paraphrased sentence 2]
**Strictly follow these guidelines to ensure that paraphrased outputs are both semantically accurate and class-aligned.**
Input S: "{text}"
Class C: "{target_class}"
"""

    def generate_neighborhood(self, text, class_names=["Class 0", "Class 1"], n_samples=50, max_iter=5):
        """
        Implements 'Algorithm 1: Local Boundary Descriptor' from the paper.
        """
        # 1. Initialization
        init_prob = self.predict_fn([text])
        orig_label_idx = np.argmax(init_prob)
        opp_label_idx = 1 - orig_label_idx

        # We need to map integer indices to string class names for the prompt
        label_map = {0: class_names, 1: class_names[1]}

        neighborhood = set()

        def parse_response(res):
            # Parse lines starting with '$' as per prompt instructions
            return [line.strip().replace('$', '').strip() for line in res.split('\n') if '$' in line]

        # 2. Initial Generation (for both classes)
        # "LLiMe prompts the LLM explicitly to generate sentences aligned with specific classes"
        for l_idx in class_names:
            prompt = self._get_neighbor_prompt(text, label_map[l_idx])
            resp = self._call_llm(prompt)
            neighborhood.update(parse_response(resp))

        # 3. Iterative Expansion (The Active Loop)
        for i in range(max_iter):
            unique_neighs = list(neighborhood)
            if not unique_neighs: break

            # Check balance
            probs = self.predict_fn(unique_neighs)
            preds = np.argmax(probs, axis=1)

            # Count how many neighbors belong to the opposite class
            opp_count = np.sum(preds == opp_label_idx)

            # Terminate if we have enough counterfactuals (balanced neighborhood)
            if opp_count >= n_samples // 2 or len(unique_neighs) >= n_samples:
                break

            # 4. Select the "Hardest" Sample to pivot
            # Paper: "selects the neighbor maximizing probability of the underrepresented class"
            # We want more of the opposite class, so we find the sample currently
            # classified as Original Class but with the highest probability of Opposite Class
            # (i.e., closest to the boundary)
            target_probs = probs[:, opp_label_idx]

            # Filter for samples that are still predicted as original class
            orig_class_mask = preds == orig_label_idx
            if not np.any(orig_class_mask):
                break  # All samples are already flipped? stop.

            candidates = np.array(unique_neighs)[orig_class_mask]
            candidate_probs = target_probs[orig_class_mask]

            # Sample closest to boundary (highest prob of target class without flipping)
            best_idx = np.argmax(candidate_probs)
            seed_text = candidates[best_idx]

            # 5. Re-prompt with new seed
            prompt = self._get_neighbor_prompt(seed_text, label_map[opp_label_idx])
            resp = self._call_llm(prompt)
            new_samples = parse_response(resp)
            neighborhood.update(new_samples)

        return list(neighborhood)

    def explain(self, text, n_samples=50):
        """
        Implements 'Relevant Words Selector' (Section 4.1.2)
        Returns: feature importance dictionary
        """
        # 1. Generate Neighborhood
        neighbors = self.generate_neighborhood(text, n_samples=n_samples)
        if not neighbors:
            return {}

        all_texts = [text] + neighbors

        # 2. Vectorize (Expanded Vocabulary)
        # LLiMe creates dictionary from Input + Neighbors
        X = self.vectorizer.fit_transform(all_texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # 3. Get Black-box predictions
        Y_probs = self.predict_fn(all_texts)
        # We explain the class predicted for the original text
        target_class = np.argmax(Y_probs)
        # Regression target: probability of the predicted class
        Y_target = Y_probs[:, target_class]

        # 4. Compute Weights (Exponential Kernel)
        # "use an exponential kernel with parameter sigma = 0.5" (Section 4.1.2)
        # Distance metric: Cosine distance on binary vectors
        D = pairwise_distances(X, X.reshape(1, -1), metric='cosine').ravel()
        sigma = 0.5
        weights = np.exp(-(D ** 2) / (sigma ** 2))

        # 5. Train Surrogate (Weighted Linear Regression)
        # Paper mentions "Linear Regression" in preliminaries and "Logistic" later.
        # Standard LIME uses Ridge (Linear) for probability regression.
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(X, Y_target, sample_weight=weights)

        # 6. Extract Feature Importance
        # Map coefficients back to words
        importance_map = {}
        input_tokens = set(self.vectorizer.inverse_transform(X.reshape(1, -1)))

        for name, coef in zip(feature_names, surrogate.coef_):
            # For ROC/PR evaluation, we typically want the absolute magnitude
            # indicating "importance".
            # If you need directional importance (supports vs opposes), keep sign.
            # LLiMe paper filters based on sign alignment, but for general benchmarking
            # against ground truth rationales, absolute importance is standard.
            importance_map[name] = abs(coef)

        return importance_map


def main():
    dataset = "sst2"
    text = "The movie was great!"

    client = openai.OpenAI(api_key=os.getenv("OPENAI"))

    model_path = TASK_MODELS[dataset]
    _, _, class_names, classifier_fn = load_model(model_path)

    # Get prediction
    probs = classifier_fn([text])[0]
    pred_idx = int(np.argmax(probs))
    predicted_label = class_names[pred_idx]

    predicted_label = class_names[pred_idx]

    # Usage in LLiMe
    llime = LLiMe(classifier_fn=classifier_fn, openai_client=client, model="gpt-4-0613")

    importance_map = llime.explain(text=text, n_samples=1)

    print(importance_map)


if __name__ == "__main__":
    main()
