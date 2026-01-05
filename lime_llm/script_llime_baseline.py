#!/usr/bin/env python3
"""
Local Boundary Descriptor - LLiMe Implementation
Generates semantically meaningful neighborhood sentences using LLMs.
Reference: Algorithm 1 from "LLiMe: enhancing text classifier explanations with LLMs"
"""

from __future__ import annotations
import os
import numpy as np
from typing import Callable
from dataclasses import dataclass
import re
from dotenv import load_dotenv
from lime_llm.utils import load_model
from lime_llm.constants import TASK_MODELS

load_dotenv()


@dataclass
class LBDConfig:
    """Configuration for Local Boundary Descriptor."""
    n: int = 10              # Desired neighbors per class
    model: str = "gpt-4o"    # LLM model
    temperature: float = 1.0  # Generation temperature

    @property
    def np(self) -> int:
        """Sentences per LLM prompt."""
        return min(10, self.n)

    @property
    def max_iter(self) -> int:
        """Maximum refinement iterations."""
        return self.n // self.np + 5


class LocalBoundaryDescriptor:
    """
    Generates meaningful neighborhood sentences using LLMs.
    Implements Algorithm 1: iterative boundary-aware neighbor generation.
    """

    PROMPT_GENERATE = """You are an advanced, context-aware paraphrasing system with a precise understanding of semantics. Your task is to generate paraphrased sentences for a given input sentence, "S", ensuring alignment with a provided class "C".

### GUIDELINES:
Generate {np} paraphrased sentences that align with the semantics of the provided class.
Each paraphrased sentence must:
- Preserve the structure of the original sentence while adapting to the designated class.
- Retain as much of the original vocabulary as possible unless modifications are necessary for semantic alignment.
- Begin with the "$" symbol for clear separation.

### INPUT:
- S: "{sentence}"
- C: "{target_class}"

### OUTPUT:
$ [Paraphrased sentence 1]
$ [Paraphrased sentence 2]
...

**Strictly follow these guidelines to ensure that paraphrased outputs are both semantically accurate and class-aligned.**"""

    PROMPT_EXCLUDE = """Generate a sentence similar to: "{sentence}"
But WITHOUT using the word: "{word}"
Output exactly one sentence starting with $:"""

    def __init__(
        self,
        classifier: Callable[[list[str]], list[float]],
        class_labels: tuple[str, str],
        config: LBDConfig | None = None,
        api_key: str | None = None
    ):
        """
        Args:
            classifier: Function mapping sentences -> P(positive_class) scores
            class_labels: (negative_class_name, positive_class_name)
            config: LBD configuration
            api_key: OpenAI API key (uses env var if None)
        """
        self.classifier = classifier
        self.neg_label, self.pos_label = class_labels
        self.config = config or LBDConfig()
        self._init_client(api_key)

    def _init_client(self, api_key: str | None):
        """Initialize OpenAI client."""
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def _query_llm(self, prompt: str) -> str:
        """Query LLM and return response text."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content

    def _parse_sentences(self, response: str) -> list[str]:
        """Extract $-prefixed sentences from LLM response."""
        sentences = []
        for line in response.split('\n'):
            if line.strip().startswith('$'):
                # Remove $ prefix and optional brackets
                text = re.sub(r'^\$\s*\[?|\]$', '', line.strip()).strip()
                if text:
                    sentences.append(text)
        return sentences

    def _generate_for_class(self, sentence: str, target_class: str) -> list[str]:
        """Generate neighbors targeting a specific class."""
        prompt = self.PROMPT_GENERATE.format(
            np=self.config.np, sentence=sentence, target_class=target_class
        )
        return self._parse_sentences(self._query_llm(prompt))

    def _generate_excluding_word(self, sentence: str, word: str) -> str | None:
        """Generate a variant sentence without a specific word."""
        prompt = self.PROMPT_EXCLUDE.format(sentence=sentence, word=word)
        results = self._parse_sentences(self._query_llm(prompt))
        return results[0] if results else None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Extract lowercase words from text."""
        return set(re.findall(r'\b\w+\b', text.lower()))

    def generate(self, sentence: str) -> tuple[list[str], dict[str, float]]:
        """
        Generate neighborhood sentences spanning the decision boundary.

        Args:
            sentence: Input sentence to build neighborhood around

        Returns:
            (neighbors, probabilities): Neighbor list and P(positive) for each
        """
        n, max_iter = self.config.n, self.config.max_iter

        # Step 1: Initial generation for both classes
        neighbors = set(
            self._generate_for_class(sentence, self.pos_label) +
            self._generate_for_class(sentence, self.neg_label)
        )
        if not neighbors:
            return [], {}

        # Track all probabilities
        probs: dict[str, float] = dict(zip(neighbors, self.classifier(list(neighbors))))

        # Partition by predicted class (threshold = 0.5)
        pos_set = {s for s, p in probs.items() if p >= 0.5}
        neg_set = {s for s, p in probs.items() if p < 0.5}

        # Step 2: Iteratively refine until n samples per class
        for _ in range(max_iter):
            if len(pos_set) >= n and len(neg_set) >= n:
                break

            new_sentences = []

            # Generate more positive-class neighbors using most positive reference
            if len(pos_set) < n and probs:
                ref_pos = max(probs, key=probs.get)
                new_sentences.extend(self._generate_for_class(ref_pos, self.pos_label))

            # Generate more negative-class neighbors using most negative reference
            if len(neg_set) < n and probs:
                ref_neg = min(probs, key=probs.get)
                new_sentences.extend(self._generate_for_class(ref_neg, self.neg_label))

            # Classify and add novel sentences
            novel = [s for s in set(new_sentences) if s not in probs]
            if not novel:
                continue

            for s, p in zip(novel, self.classifier(novel)):
                probs[s] = p
                (pos_set if p >= 0.5 else neg_set).add(s)

        # Step 3: Select top-n most polarized from each class
        ranked = sorted(probs, key=probs.get, reverse=True)
        final = set(ranked[:n] + ranked[-n:])  # Top n positive + top n negative

        # Step 4: Add exclusion variants for words appearing in ALL neighbors
        input_words = self._tokenize(sentence)
        neighbor_word_sets = [self._tokenize(s) for s in final]

        if neighbor_word_sets:
            common_words = input_words.intersection(*neighbor_word_sets)
            for word in common_words:
                excluded = self._generate_excluding_word(sentence, word)
                if excluded and excluded not in probs:
                    probs[excluded] = self.classifier([excluded])[0]
                    final.add(excluded)

        result = list(final)
        return result, {s: probs[s] for s in result}


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":

    dataset = "sst2"
    text = "The movie was great!"

    model_path = TASK_MODELS[dataset]
    _, _, class_names, classifier_fn = load_model(model_path)
    probs = classifier_fn([text])[0]
    pred_idx = int(np.argmax(probs))

    def mock_classifier(sentences: list[str]) -> list[float]:
        """Mock classifier: Sports (positive) vs Entertainment (negative)."""
        sports_kw = {"sports", "game", "team", "player", "win", "score", "match", "coach"}
        entertain_kw = {"music", "song", "concert", "band", "movie", "show", "album", "actor"}

        results = []
        for s in sentences:
            words = set(s.lower().split())
            sp = len(words & sports_kw)
            en = len(words & entertain_kw)
            prob = sp / (sp + en) if (sp + en) > 0 else 0.5
            results.append(prob)
        return results

    # Initialize
    config = LBDConfig(n=5, model="gpt-4o-mini")
    lbd = LocalBoundaryDescriptor(
        classifier=mock_classifier,
        class_labels=("Entertainment", "Sports"),
        config=config,
        api_key=os.getenv("OPENAI")
    )

    # Generate neighbors
    test_sentence = "The team played an amazing game last night."
    neighbors, probs = lbd.generate(test_sentence)

    # Display results
    print(f"Input: {test_sentence}")
    print(f"Predicted class: {'Sports' if mock_classifier([test_sentence])[0] >= 0.5 else 'Entertainment'}")
    print(f"\nGenerated {len(neighbors)} neighbors:\n")

    for s in sorted(neighbors, key=lambda x: probs[x], reverse=True):
        label = "Sports" if probs[s] >= 0.5 else "Entertainment"
        print(f"  [{probs[s]:.3f}] ({label:13}) {s}")
