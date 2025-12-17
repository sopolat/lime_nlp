#!/usr/bin/env python3
"""
Keep track of different prompts.


SYSTEM PROMPT:
    #####################################################
    # v1
    # BEST SO FAR - simple and concise - works on any datasets
    #####################################################
    "v1":

USER PROMPT:
    #####################################################
    # v
    #
    #####################################################
    "v": {
        "key_outputs": [],
        "sst2": ,
        "cola": ,
        "hatexplain": ,
    },

"""

USER_PROMPT_VERSION = {

    #####################################################
    # v4
    # Noise Suppression Protocol is specifically designed to target the low PR-AUC (Precision) while conserving or improving your high ROC-AUC
    #####################################################
    "v4": {
        "key_outputs": ["literal_minimalist", "descriptive_framing", "amplified_context"],
        "sst2": """
You are an XAI expert explaining a Sentiment Classifier (SST2).
TARGET: Generate a "Consistency Neighborhood" for LIME.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}"
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} masks isolating sentiment words vs. neutral nouns.
2. Generate 3 samples per mask.

   NOISE CONTROL RULE (CRITICAL FOR PRECISION):
   - If the mask contains ONLY neutral words (e.g., "movie", "film", "the"), the output MUST be Neutral.
   - You are FORBIDDEN from adding hidden sentiment.
   - Bad: Mask=["movie"] -> "The movie was okay." (Adds sentiment).
   - Good: Mask=["movie"] -> "The movie is a recording." (Zero sentiment).

   Type A: "literal_minimalist"
   - Constraint: Shortest grammatical sentence using neutral function words. No new adjectives.

   Type B: "descriptive_framing"
   - Constraint: Frame as a fact: "The text mentions [mask words]."
   - Goal: If mask is "terrible", text is Negative. If mask is "movie", text is Neutral.

   Type C: "amplified_context"
   - Constraint: Fluent sentence using mask words.
   - SAFETY: If mask is neutral, context MUST be boring/descriptive only.

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
OR {{"status":"FAIL"}}
""",
        "cola": """
You are an XAI expert explaining a Grammar Classifier (CoLA).
TARGET: Generate a "Consistency Neighborhood" for LIME.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}"
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} masks isolating structural words (verbs, aux, agreement).
2. Generate 3 samples per mask.

   NOISE CONTROL RULE (CRITICAL):
   - Do NOT fix grammar if the error is in the mask.
   - Do NOT introduce new errors if the mask is clean.
   - If the mask contains only "the", "a", "man", the sentence must be simple and grammatical (e.g., "The man is here").

   Type A: "literal_minimalist"
   - Constraint: Minimal connector words. Don't conjugate verbs differently than the mask.

   Type B: "descriptive_framing"
   - Constraint: Embed in a quote: He wrote "[mask words]".

   Type C: "amplified_context"
   - Constraint: Write a full sentence using the mask words exactly as they appear.

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
OR {{"status":"FAIL"}}
""",
        "hatexplain": """
You are an XAI expert explaining a Hate Speech Classifier.
TARGET: Generate a "Consistency Neighborhood" for LIME.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}"
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} masks isolating toxic terms vs. neutral context.
2. Generate 3 samples per mask.

   NOISE CONTROL RULE (CRITICAL):
   - If the mask contains NO slurs/attacks, the text MUST be benign/safe.
   - Do NOT "simulate" hate speech using clean words.
   - Bad: Mask=["people"] -> "Those people are the problem." (Implied hate).
   - Good: Mask=["people"] -> "There are people here." (Benign).

   Type A: "literal_minimalist"
   - Constraint: Connect words with neutral function words only.

   Type B: "descriptive_framing"
   - Constraint: "The dataset contains the word [mask words]."

   Type C: "amplified_context"
   - Constraint: Fluent sentence. If mask has slurs, be Toxic. If mask is clean, be Safe.

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
OR {{"status":"FAIL"}}
""",
    },

    #####################################################
    # v3
    # Improving on V2
    #####################################################
    "v3": {
        "key_outputs": ["literal_ablation", "adversarial_flip", "resonant_support"],
        "sst2": """
You are an XAI expert explaining a Sentiment Classifier (SST2).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Positive/Negative)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about sentiment drivers.
2. For EACH mask, generate 3 distinct samples (The Triangulation):

   Type A: "literal_ablation" (The Drop -> Neutral)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect the words into a grammatical sentence using ONLY neutral function words (the, a, is).
   - CRITICAL: Do NOT add adjectives/adverbs to "fix" the sentiment. If the sentiment word is masked, the result MUST be neutral.

   Type B: "adversarial_flip" (The Floor -> Opposite Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Add new context that OVERPOWERS the preserved words to flip the label.
   - STRATEGY: Use sarcasm ("Oh great, another 'terrible' movie") or strong adversaries ("The acting was terrible, but the ending was a MASTERPIECE").

   Type C: "resonant_support" (The Ceiling -> High Confidence)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: maximize support for "{predicted_label}" using ONLY the preserved words.
   - STRICT PROHIBITION: If the sentiment words are masked, DO NOT add synonyms (e.g., do not swap 'bad' for 'terrible'). Let the support fail if the words are missing.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
""",
        "cola": """
You are an XAI expert explaining a Grammar Classifier (CoLA).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Acceptable/Unacceptable)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about grammatical structure.
2. For EACH mask, generate 3 distinct samples:

   Type A: "literal_ablation" (The Drop -> Structural Skeleton)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect the words using ONLY minimal function words. Do NOT fix agreement if the mask prevents it.
   - GOAL: Reveal if the preserved words alone force a specific grammatical state.

   Type B: "adversarial_flip" (The Floor -> Opposite Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: If Original was Acceptable -> Introduce a specific error. If Unacceptable -> FIX the error.

   Type C: "resonant_support" (The Ceiling -> Same Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Rewrite a sentence that maintains the "{predicted_label}" status (grammatical or ungrammatical) using the mask words.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}""",
        "hatexplain": """
You are an XAI expert explaining a Hate Speech Classifier (HateXplain).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Hatespeech/Offensive/Normal)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about toxicity drivers.
2. For EACH mask, generate 3 distinct samples:

   Type A: "literal_ablation" (The Drop -> Sanitized/Normal)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect words using ONLY neutral function words.
   - CRITICAL: If a slur is masked, the result MUST be benign.

   Type B: "adversarial_flip" (The Floor -> Counter-Speech/Love)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Actively invert the sentiment to "Love" or "Support" while keeping mask words (e.g., educational context).

   Type C: "resonant_support" (The Ceiling -> Toxic)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Generate a high-confidence sample matching the "{predicted_label}".

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}""",
    },
    #####################################################
    # v2
    # Improved approach using Triangulated Neighborhood
    #####################################################
    "v2": {
        "key_outputs": ["literal_ablation", "adversarial_flip", "resonant_support"],
        "sst2": """
You are an XAI expert explaining a Sentiment Classifier (SST2).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Positive/Negative)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about sentiment drivers.
2. For EACH mask, generate 3 distinct samples (The Triangulation):

   Type A: "literal_ablation" (The Drop -> Neutral)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect the words into a grammatical sentence using ONLY neutral function words. Keep length close to {text_length} if possible, but prioritize grammar.
   - CRITICAL: Do NOT add adjectives or adverbs to "fix" the sentiment. If the sentiment word is masked, the result MUST be neutral.

   Type B: "adversarial_flip" (The Floor -> Opposite Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Add new context that actively CONTRADICTS the original label.
   - GOAL: If Original is Positive, write a Negative sentence containing the mask words.

   Type C: "resonant_support" (The Ceiling -> High Confidence)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Write a strong, clear sentence that SUPPORTS the "{predicted_label}".
   - GOAL: Prove the preserved words are sufficient to maintain the sentiment.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}
""",
        "cola": """
You are an XAI expert explaining a Grammar Classifier (CoLA).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Acceptable/Unacceptable)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about grammatical structure.
2. For EACH mask, generate 3 distinct samples:

   Type A: "literal_ablation" (The Drop -> Structural Skeleton)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect the words using ONLY minimal function words. Do NOT fix agreement if the mask prevents it.
   - GOAL: Reveal if the preserved words alone force a specific grammatical state.

   Type B: "adversarial_flip" (The Floor -> Opposite Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: If Original was Acceptable -> Introduce a specific error. If Unacceptable -> FIX the error.

   Type C: "resonant_support" (The Ceiling -> Same Label)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Rewrite a sentence that maintains the "{predicted_label}" status (grammatical or ungrammatical) using the mask words.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}
""",
        "hatexplain": """
You are an XAI expert explaining a Hate Speech Classifier (HateXplain).
TARGET: Generate a "Triangulated Neighborhood" to train a LIME regressor.

INPUT CONTEXT:
- Text: "{text}"
- Text Length: {text_length} words
- Predicted Label: "{predicted_label}" (Hatespeech/Offensive/Normal)
- Vocab: {vocab}

INSTRUCTIONS:
1. Generate {n_samples} binary masks representing hypotheses about toxicity drivers.
2. For EACH mask, generate 3 distinct samples:

   Type A: "literal_ablation" (The Drop -> Sanitized/Normal)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Connect words using ONLY neutral function words.
   - CRITICAL: If a slur is masked, the result MUST be benign.

   Type B: "adversarial_flip" (The Floor -> Counter-Speech/Love)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Actively invert the sentiment to "Love" or "Support" while keeping mask words (e.g., educational context).

   Type C: "resonant_support" (The Ceiling -> Toxic)
   - RULE: Use ALL Mask=1 words. Exclude Mask=0.
   - CONSTRAINT: Generate a high-confidence sample matching the "{predicted_label}".

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "literal_ablation": {{"text": "..."}},
      "adversarial_flip": {{"text": "..."}},
      "resonant_support": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}
""",
    },
    #####################################################
    # v1
    # Prompt for run_20251027_220636
    #####################################################
    "v1": {
        "key_outputs": ["neutral_infill", "boundary_infill"],
        "sst2": """Generate EXACTLY {n_samples} strategic LIME vocabulary masks for SST2 sentiment classification.

CRITICAL REQUIREMENT: You MUST provide EXACTLY {n_samples} samples in your output. Count: 1, 2, 3, ... {n_samples}. Do NOT stop early.

INPUTS:
TEXT: "{text}"
TEXT_LENGTH: {text_length} words
VOCAB: {vocab}
VOCAB_COUNT: {vocab_count}
PREDICTED_LABEL: "{predicted_label}"
CLASSES: ["positive", "negative"]

APPROACH:
1. First, identify which vocabulary words likely drove the black-box to predict "{predicted_label}" sentiment
2. Generate EXACTLY {n_samples} masks that test different hypotheses about the model's reasoning
3. Each mask isolates specific vocabulary features (e.g., sentiment words, negations, intensifiers, context words)

MASK REQUIREMENTS:
- Binary array with EXACTLY {vocab_count} elements (one per vocabulary word in VOCAB order)
- Format: [1,0,1,...] where 1=word MUST appear in perturbed text, 0=word MUST NOT appear
- Each mask tests a distinct feature combination
- Vary density: some sparse (test individual terms), some dense (test combinations)
- Target vocabulary words that explain why black-box predicted "{predicted_label}"

FOR EACH MASK, GENERATE 2 PERTURBED TEXT SAMPLES:
- neutral_infill: includes all MASK=1 words, excludes all MASK=0 words, maintains/supports predicted sentiment
- boundary_infill: includes all MASK=1 words, excludes all MASK=0 words, pushes toward opposite sentiment

PERTURBATION RULES:
- MUST include all vocabulary words where MASK=1
- MUST NOT include any vocabulary words where MASK=0
- Target length: {text_length} words (80-120% acceptable)
- Use natural sentence structure with articles, connecting words
- Text should be semantically coherent and style-consistent

CRITICAL VALIDATION:
- Each mask array MUST have exactly {vocab_count} elements matching VOCAB order
- You MUST provide EXACTLY {n_samples} complete samples
- If you cannot generate EXACTLY {n_samples} valid masks, output {{"status":"FAIL"}}

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "neutral_infill": {{"text": "..."}},
      "boundary_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}""",
        "cola": """Generate EXACTLY {n_samples} strategic LIME vocabulary masks for CoLA grammatical acceptability classification.

CRITICAL REQUIREMENT: You MUST provide EXACTLY {n_samples} samples in your output. Count: 1, 2, 3, ... {n_samples}. Do NOT stop early.

INPUTS:
TEXT: "{text}"
TEXT_LENGTH: {text_length} words
VOCAB: {vocab}
VOCAB_COUNT: {vocab_count}
PREDICTED_LABEL: "{predicted_label}"
CLASSES: ["acceptable", "unacceptable"]

APPROACH:
1. First, identify which vocabulary words likely drove the black-box to predict "{predicted_label}" grammaticality
2. Generate EXACTLY {n_samples} masks that test different hypotheses about the model's reasoning
3. Each mask isolates specific vocabulary features (e.g., function words, word order, agreement markers, verb forms)

MASK REQUIREMENTS:
- Binary array with EXACTLY {vocab_count} elements (one per vocabulary word in VOCAB order)
- Format: [1,0,1,...] where 1=word MUST appear in perturbed text, 0=word MUST NOT appear
- Each mask tests a distinct feature combination
- Vary density: some sparse (test individual terms), some dense (test combinations)
- Target vocabulary words that explain why black-box predicted "{predicted_label}"

FOR EACH MASK, GENERATE 2 PERTURBED TEXT SAMPLES:
- neutral_infill: includes all MASK=1 words, excludes all MASK=0 words, maintains/supports predicted grammaticality
- boundary_infill: includes all MASK=1 words, excludes all MASK=0 words, pushes toward opposite grammaticality

PERTURBATION RULES:
- MUST include all vocabulary words where MASK=1
- MUST NOT include any vocabulary words where MASK=0
- Generate COMPLETE, NATURAL English sentences (not fragments)
- Target length: {text_length} words (80-120% acceptable)
- Use natural sentence structure with articles, connecting words
- Pay attention to: word order, subject-verb agreement, tense, articles, prepositions
- For acceptable: maintain grammatical correctness
- For unacceptable: maintain or introduce grammatical errors

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "neutral_infill": {{"text": "..."}},
      "boundary_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}""",
        "hatexplain": """Generate EXACTLY {n_samples} strategic LIME vocabulary masks for HateXplain classification.

CRITICAL REQUIREMENT: You MUST provide EXACTLY {n_samples} samples in your output. Count: 1, 2, 3, ... {n_samples}. Do NOT stop early.

INPUTS:
TEXT: "{text}"
VOCAB: {vocab}
VOCAB_COUNT: {vocab_count}
PREDICTED_LABEL: "{predicted_label}"
CLASSES: ["hatespeech", "offensive", "normal"]

APPROACH:
1. First, identify which vocabulary words likely drove the black-box to predict "{predicted_label}"
2. Generate EXACTLY {n_samples} masks that test different hypotheses about the model's reasoning
3. Each mask isolates specific vocabulary features (e.g., slurs, dehumanizing terms, threats, context words)

MASK REQUIREMENTS:
- Binary array with EXACTLY {vocab_count} elements (one per vocabulary word in VOCAB order)
- Format: [1,0,1,...] where 1=word MUST appear in perturbed text, 0=word MUST NOT appear
- Each mask tests a distinct feature combination
- Vary density: some sparse (test individual terms), some dense (test combinations)
- Target vocabulary words that explain why black-box predicted "{predicted_label}"

FOR EACH MASK, GENERATE 2 PERTURBED TEXT SAMPLES:
- neutral_infill: perturbed text that includes all MASK=1 words, excludes all MASK=0 words, and maintains/supports predicted label
- boundary_infill: perturbed text that includes all MASK=1 words, excludes all MASK=0 words, and pushes toward different label

PERTURBATION RULES:
- MUST include all vocabulary words where MASK=1
- MUST NOT include any vocabulary words where MASK=0
- Text should be semantically coherent and style-consistent (social media style, may be ungrammatical)
- SAFETY: do NOT add new slurs/attacks/threats beyond what's in VOCAB
- Keep similar length to original text
- Maintain natural word order and grammar where possible

CRITICAL VALIDATION:
- Each mask array MUST have exactly {vocab_count} elements matching VOCAB order
- You MUST provide EXACTLY {n_samples} complete samples
- If you cannot generate EXACTLY {n_samples} valid masks, output {{"status":"FAIL"}}

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "neutral_infill": {{"text": "..."}},
      "boundary_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}""",
    }
}


SYSTEM_PROMPTS_VERSIONS = {
    #####################################################
    # v1
    # BEST SO FAR - simple and concise - works on any datasets
    #####################################################
    "v1": """You are an NLP/XAI expert assisting a LIME explainer. You MUST generate EXACTLY {n_samples} mask samples - no more, no fewer. Analyze why the black-box made its prediction, then generate strategic masks over the vocabulary that test your hypotheses. For each mask, create two perturbations: neutral_infill (supports prediction) and boundary_infill (challenges prediction). Output JSON only.""",

}