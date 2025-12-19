#!/usr/bin/env python3
"""
Keep track of different prompts.


SYSTEM PROMPT:
    #####################################################
    # v
    #
    #####################################################
    "v":

USER PROMPT:
    #####################################################
    # v
    #
    #####################################################
    "v":

"""

# Need to have "key_outputs" and "user_prompt"
################################## USER_PROMPT_VERSIONS ##################################
USER_PROMPT_VERSIONS = {
    #####################################################
    # v9 - Include Protocol in Data Description
    #####################################################
    "v9": {
        "key_outputs": ["neutral_infill", "contrastive_infill"],
        "user_prompt": """# INPUT
Text: "{text}"
Label: "{predicted_label}"
Vocab: {vocab} (length={vocab_count})
Samples needed: {n_samples}

{dataset_description}

# OUTPUT FORMAT (JSON)
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "neutral_infill": {{"text": "..."}},
      "contrastive_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status": "FAIL"}}"""
    },

    #####################################################
    # v8 - Improved V4 - which was missing descriptions
    #####################################################
    "v8": {
        "key_outputs": ["neutral_infill", "boundary_infill"],
        "user_prompt": """# INPUT
TEXT: "{text}"
PREDICTED_LABEL: "{predicted_label}"
VOCAB (ordered): {vocab}
VOCAB_COUNT: {vocab_count}
N_SAMPLES: {n_samples}

# TASK RULES (dataset-specific)
{dataset_description}

# HARD CONSTRAINTS (MUST FOLLOW)
- Output EXACTLY N_SAMPLES samples.
- Each mask is a binary list of length VOCAB_COUNT (0/1 only).
- masked_vocab MUST be exactly: [vocab[i] for i where mask[i]==1] (same order, no extras).
- Generated text may use ONLY:
  (a) words in masked_vocab, PLUS
  (b) function words explicitly allowed by TASK RULES (if any).
- Do NOT introduce synonyms/antonyms, new sentiment/toxic/grammar tokens, new names, or extra content words.

# NEIGHBORHOOD DESIGN (diverse + informative)
Create a balanced set of masks that probes local behavior:
1) Full mask (all 1s): 1 sample.
2) Leave-one-out: masks that drop exactly 1 vocab word (use several different dropped words).
3) Small “Signal-focused” masks: keep only likely Signal words (and minimal glue if allowed).
4) Noise-only masks: keep only likely Noise words (per task rules).
5) Interaction masks: small pairs/triples that test combinations (e.g., negation+adjective, error token+context).

# INFILL GENERATION (for each mask)
Generate TWO texts using ONLY allowed words:

A) neutral_infill (Label-preserving)
- Must satisfy TASK RULES for this dataset.
- If mask contains Signal words, the text MUST reflect "{predicted_label}".
- If mask contains ONLY Noise words, the text MUST be neutral/factual (zero sentiment/toxicity).

B) boundary_infill (Decision-boundary probing)
- Use the SAME allowed words as neutral_infill (masked_vocab + allowed function words only).
- Push prediction toward the opposite label by:
  - re-ordering / changing scope,
  - using negation/hedging ONLY if allowed,
  - contrastive framing with allowed glue words,
  - if a true flip is impossible under constraints, make it maximally ambiguous to reduce confidence.
- Must stay grammatical and natural (no word salad).

# OUTPUT FORMAT (JSON)
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
OR {{"status": "FAIL", "reason": "..."}}"""
    },
    #####################################################
    # v7 - Compact Triple-Infill (Variance-Optimized)
    # - 3 samples: primary, contrastive, minimal
    # - Task definitions inline, concise system prompt
    # - Fixes SST2 sparsity while maintaining CoLA/HateXplain
    #####################################################
    "v7": {
        "key_outputs": ["primary_sample", "contrastive_sample", "minimal_sample"],
        "user_prompt": """# INPUT
Text: "{text}"
Label: "{predicted_label}"
Vocab: {vocab} ({vocab_count} words)
Samples: {n_samples}

# TASK
{dataset_description}

# PROTOCOL
1. Generate {n_samples} masks (mix small/medium/large sizes)
2. Per mask, create 3 samples using ONLY masked words:
   - primary_sample (per task rules)
   - contrastive_sample (per task rules)
   - minimal_sample (per task rules)

# OUTPUT FORMAT (JSON)
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "primary_sample": {{"text": "..."}},
      "contrastive_sample": {{"text": "..."}},
      "minimal_sample": {{"text": "..."}}
    }}
  ]
}}
OR {{"status": "FAIL", "reason": "..."}}"""
    },
    #####################################################
    # v6 - Minimal Phrasing Strategy (Variance-Optimized)
    # - Simple, fragment-based infills for all tasks
    # - Prioritizes model variance over linguistic fluency
    # - Reduces sparsity by testing words directly
    #####################################################
    "v6": {
        "key_outputs": ["sample_text"],
        "user_prompt": """# INPUT
Text: "{text}"
Label: "{predicted_label}"
Vocab: {vocab} (length={vocab_count})
Samples needed: {n_samples}

# TASK RULES
{dataset_description}

# PROTOCOL
1. Generate {n_samples} binary masks following task strategies
2. For each mask, create MINIMAL text using ONLY masked words:
   - Use masked words with minimal function words (a, the, is, are)
   - Keep it SHORT and SIMPLE (2-8 words typical)
   - Fragments and incomplete sentences are OK
   - Avoid elaborate or overly fluent constructions
3. Goal: Test individual word importance directly

# OUTPUT (JSON)
{{
  "status": "OK",
  "samples": [
    {{
      "mask": [1,0,1,...],
      "masked_vocab": ["w1","w2",...],
      "mask_strategy": "strategy_name",
      "sample_text": "minimal phrasing here"
    }}
  ]
}}"""
    },

    #####################################################
    # v5 - Simplified SST2 Strategy (50% Random + Natural Phrasing)
    # - CoLA/HateXplain: Keep v4 dual infill (working great)
    # - SST2: Single infill + 50% random masks (avoid contradictions)
    #####################################################
    "v5": {
        "key_outputs": ["neutral_infill", "contrastive_infill"],  # contrastive optional per task
        "user_prompt": """# INPUT
Text: "{text}"
Label: "{predicted_label}"
Vocab: {vocab} (length={vocab_count})
Samples needed: {n_samples}

# TASK RULES
{dataset_description}

# PROTOCOL
1. Generate {n_samples} binary masks [{vocab_count} values] following task strategies
2. For each mask, create infills using ONLY masked words (see task rules for count)
3. Use only masked_vocab + allowed function words (if specified)

# OUTPUT FORMAT (JSON)
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "neutral_infill": {{"text": "..."}},
      "contrastive_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status": "FAIL", "reason": "..."}}"""
    },
    #####################################################
    # v4 - Task-Adaptive Infill Strategy
    # - Task-specific infill types (weak/boundary/clinical)
    # - Concrete mask strategies with percentages
    # - Strict vocab constraints to prevent off-manifold samples
    # - Addresses v3's PR-AUC collapse on compositional tasks
    #####################################################
    "v4": {
        "key_outputs": ["neutral_infill", "contrastive_infill"],
        "user_prompt": """# INPUT
Text: "{text}"
Label: "{predicted_label}"
Vocab: {vocab} (length={vocab_count})
Samples needed: {n_samples}

# TASK RULES
{dataset_description}

# PROTOCOL
1. Generate {n_samples} binary masks [{vocab_count} values] following task strategies
2. For each mask, create TWO infills using ONLY masked words:
   - neutral_infill: as defined in task rules
   - contrastive_infill: as defined in task rules
3. Use only masked_vocab + allowed function words (if specified)

# OUTPUT FORMAT (JSON)
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "masked_vocab": ["word1", "word2", ...],
      "neutral_infill": {{"text": "..."}},
      "boundary_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status": "FAIL", "reason": "..."}}"""
    },
    #####################################################
    # v3 - Hypothesis-Driven Dual Infill (Protocol Included)
    #####################################################
    "v3": {
        "key_outputs": ["neutral_infill", "boundary_infill"],
        "user_prompt": """# INPUT CONTEXT
- Text: "{text}"
- Predicted Label: "{predicted_label}"
- Vocab List: {vocab}
- Vocab Count: {vocab_count} (Masks must have exactly this length)
- Required Samples: {n_samples}

# TASK SEMANTICS
{dataset_description}

# PROTOCOL

**Step 1: Mask Generation**
Generate {n_samples} diverse binary masks [{vocab_count} values: 0 or 1].
- mask[i]=1: include vocab[i], mask[i]=0: exclude vocab[i]
- Test different hypotheses: feature importance, combinations, necessity
- Create varied masks: some with many features, some with few

**Step 2: Dual Infill Generation**
For each mask, generate TWO samples using ONLY masked vocabulary:

A. **neutral_infill** (Label-Preserving)
   - Maintain predicted label "{predicted_label}"
   - Use masked words naturally to support the prediction
   - Write fluent text that agrees with the label

B. **boundary_infill** (Label-Challenging)
   - Push toward the opposite label
   - Use masked words with contrary framing, hedging, or negation
   - Maximize semantic shift while staying grammatical

**Step 3: Quality Check**
- Both infills use ONLY words where mask[i]=1
- Both are grammatical and natural (no word salad)
- They form a contrastive pair (same vocab, different semantics)

# OUTPUT FORMAT (JSON)
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "masked_vocab": ["word1", "word2", ...],
      "neutral_infill": {{"text": "..."}},
      "boundary_infill": {{"text": "..."}}
    }}
  ]
}}
OR {{"status": "FAIL", "reason": "..."}}"""
    },
    # ==============================================================================
    # v2: CONTRASTIVE PAIR STRATEGY (Universal Template)
    # Goal: Create variance (High vs Low Prob) to fix SST2's scalar issues.
    # Constraint: Uses the FIXED Dataset Descriptions.
    # ==============================================================================
    "v2": {
        "key_outputs": ["signal_anchor", "boundary_pusher"],
        "user_prompt": """
# THE CONTRASTIVE PROTOCOL
For every mask, generate 2 opposing samples using the "Dataset Rules" provided in the input:

1. Sample A: "signal_anchor" (The Ceiling)
   - Goal: Maximize the probability of the "{predicted_label}".
   - Method: Apply the "Signal Rule" from the Dataset Rules strictly.
   - Context: Place the mask words in a clear, unambiguous statement supporting the label.

2. Sample B: "boundary_pusher" (The Floor)
   - Goal: Minimize the probability (Push toward Neutral or Opposite).
   - Method:
     - If Mask has Signal Words: DILUTE or NEGATE the signal. (e.g., Use "not [word]", "was it [word]?", or "the [word] character").
     - If Mask has Noise Words: Apply the "Noise Rule" strictly (Keep it neutral).

# INPUT CONTEXT
- Text: "{text}"
- Predicted Label: "{predicted_label}"
- Vocab List: {vocab}
- Vocab Count: {vocab_count} (Masks must have exactly this length)
- Required Samples: {n_samples}

# DATASET RULES (CRITICAL)
{dataset_description}

# INSTRUCTIONS
1. Analyze the text and the Rules to identify "Signal" vs "Noise" words.
2. Generate {n_samples} masks.
3. For EACH mask, generate exactly 2 opposing samples (Anchor vs Pusher).

# STRICT CONSTRAINTS
- **Mask Alignment:** Every mask must be a binary list with exactly {vocab_count} items.
- **Usage:** You MUST use the mask words in the text. Do not just list them.
- **Safety:** Do not use irony or sarcasm that creates ambiguity. Be direct.

OUTPUT FORMAT (JSON ONLY):
{{
  "status": "OK",
  "sample_count": {n_samples},
  "samples": [
    {{
      "mask": [1,0,1,...],
      "signal_anchor": {{"text": "..."}},
      "boundary_pusher": {{"text": "..."}}
    }}
  ]
}}
OR {{"status":"FAIL"}}"""
    },
    #####################################################
    # v1
    #
    #####################################################
    "v1": {
        "key_outputs": ["literal_minimalist", "descriptive_framing", "amplified_context"],
        "user_prompt": """# INPUT CONTEXT
- Text: "{text}"
- Length: {text_length} words
- Predicted Label: "{predicted_label}"
- Vocab List: {vocab}
- Vocab Count: {vocab_count} (Masks must have exactly this length)
- Required Samples: {n_samples}

# DATASET RULES (CRITICAL)
{dataset_description}

# INSTRUCTIONS
1. Analyze the text. Identify which words are "Signal" vs "Noise" based on the Rules above.
2. Generate {n_samples} masks. Mix Signal-focused masks and Noise-focused masks.
3. Apply the PROTOCOL (from System Prompt) to generate samples.

# STRICT CONSTRAINTS
- **Consistency:** You must strictly follow the "Signal" and "Noise" definitions provided in the Rules above.
- **Safety:** NEVER flip the label using irony, complex negation, or adversarial context.
- **Format:** Ensure every mask array has exactly {vocab_count} items.

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
OR {{"status":"FAIL"}}"""
    }
}

################################## SYSTEM_PROMPT_VERSIONS ##################################
SYSTEM_PROMPT_VERSIONS = {
    "v9": """You are an Explainable AI (XAI) expert specializing in expert generating LIME training samples.""",
    #####################################################
    # v7 - Minimal System Prompt
    #####################################################
    "v7": """XAI expert generating LIME samples.

Create 3 diverse samples per mask to maximize model prediction variance.
Follow task-specific infill rules exactly.
Use only masked vocabulary + allowed function words.""",

    #####################################################
    # v5 - Minimal Phrasing for Maximum Variance
    #####################################################
    "v5": """You are an XAI expert generating LIME training samples.

Generate MINIMAL, SIMPLE text using masked words. Your samples should maximize model prediction variance, not linguistic fluency.

Core principles:
- SHORT: Use 2-8 words typically
- SIMPLE: Avoid complex sentence structures  
- DIRECT: Test words directly, not in elaborate context
- FRAGMENTS OK: "great performance" beats "The great performance stands out"

Why: Complex fluent text creates similar model predictions across samples, reducing the linear surrogate's ability to learn feature importance.""",

    #####################################################
    # v4 - Task-Adaptive Infill Strategy
    # - Task-specific infill types (weak/boundary/clinical)
    # - Concrete mask strategies with percentages
    # - Strict vocab constraints to prevent off-manifold samples
    # - Addresses v3's PR-AUC collapse on compositional tasks
    #####################################################
    "v4": """You are an XAI expert generating LIME training samples.

Generate semantically-aware perturbations that stay on-manifold (unlike random word dropout).

Core principles:
- Use only masked vocabulary + allowed function words
- Generate natural, fluent text
- Follow task-specific strategies exactly
- Create effective contrastive pairs""",

    #####################################################
    # v3 - General XAI Expert Role (Protocol in User Prompt)
    #####################################################
    "v3": """You are an XAI expert specializing in LIME explanations for text classifiers.

# YOUR ROLE
Generate semantically-aware training samples for local surrogate models. Unlike vanilla LIME's random word dropout, your samples stay on the semantic manifold to improve explanation faithfulness and stability.

# CORE PRINCIPLES
- **Semantic Fidelity**: Generate natural, fluent text using only masked vocabulary
- **Hypothesis-Driven**: Test specific hypotheses about feature importance
- **Contrastive Power**: Create samples that probe decision boundaries effectively
- **Model Faithfulness**: Approximate the model's decision logic, not your intuitions

Follow the detailed protocol and task-specific semantics provided in each request.""",

    #####################################################
    # v2
    # Keep it general - move a lot of details in the user prompts.
    #####################################################
    "v2": """You are an XAI expert explaining a black-box model.
YOUR GOAL: Generate a "Contrastive Neighborhood" to train a LIME regressor.
LIME requires VARIANCE: It needs to see the same words in high-confidence contexts AND low-confidence contexts.

""",
    #####################################################
    # v1
    # Having a lot of info in the system
    #####################################################
    "v1": """You are an XAI (Explainable AI) expert explaining a black-box model.
YOUR GOAL: Generate a "Consistency Neighborhood" to train a LIME regressor.

# TECHNICAL CONSTRAINTS (CRITICAL)
- **Vocabulary Definition:** The provided "Vocab List" contains the UNIQUE words from the input text.
- **Mask Alignment:** Every generated mask MUST be a binary list `[1, 0, ...]` with a length EXACTLY equal to the length of the "Vocab List".
- **Mapping:** Index `i` in the mask corresponds strictly to index `i` in the "Vocab List".

# THE GENERATION PROTOCOL
For every request, you must generate binary masks and 3 specific sample types per mask.

1. Type A: "literal_minimalist"
   - Constraint: Create the shortest possible sentence using only neutral function words (the, it, is).
   - Purpose: Isolate the "pure" signal.

2. Type B: "descriptive_framing"
   - Constraint: Frame the mask words in a neutral, factual structure (e.g., "The text contains [words]").
   - Purpose: Test if words trigger the label without emotional context.

3. Type C: "amplified_context"
   - Constraint: Write a fluent sentence using the mask words.
   - Requirement: Maximize the signal. If the mask has signal words, be Strong. If the mask has noise words, be Boring.
"""
}

################################## DATASET_DESCRIPTION ##################################
DATASET_DESCRIPTION = {
    #####################################################
    # v9 - Include Protocol in Data Description
    #####################################################
    "v9": {
        "sst2": """ # YOUR ROLE
Generate semantically-aware training samples for local surrogate models. Unlike vanilla LIME's random word dropout, your samples stay on the semantic manifold to improve explanation faithfulness and stability.

# CORE PRINCIPLES
- **Semantic Fidelity**: Generate natural, fluent text using only masked vocabulary
- **Hypothesis-Driven**: Test specific hypotheses about feature importance
- **Contrastive Power**: Create samples that probe decision boundaries effectively
- **Model Faithfulness**: Approximate the model's decision logic, not your intuitions

Follow the detailed protocol and task-specific semantics provided in each request.

# TASK RULES
Task: Sentiment Analysis (Positive/Negative).
Signal: Adjectives, adverbs, and intensifiers (e.g., "terrible", "great").
Noise: Neutral nouns and function words (e.g., "movie", "film").
RULE: If the mask contains Signal words, the text MUST reflect the predicted label.
RULE: If the mask contains ONLY Noise words, the text MUST be Neutral (boring, factual, zero emotion).

# PROTOCOL
**Step 1: Mask Generation**
Generate {n_samples} diverse binary masks [{vocab_count} values: 0 or 1].
- mask[i]=1: include vocab[i], mask[i]=0: exclude vocab[i]
- Test different hypotheses: feature importance, combinations, necessity
- Create varied masks: some with many features, some with few

**Step 2: Dual Infill Generation**
For each mask, generate TWO samples using ONLY masked vocabulary:

A. **neutral_infill** (Label-Preserving)
   - Maintain predicted label "{predicted_label}"
   - Use masked words naturally to support the prediction
   - Write fluent text that agrees with the label

B. **contrastive_infill** (Label-Challenging)
   - Push toward the opposite label
   - Use masked words with contrary framing, hedging, or negation
   - Maximize semantic shift while staying grammatical

**Step 3: Quality Check**
- Both infills use ONLY words where mask[i]=1
- Both are grammatical and natural (no word salad)
- They form a contrastive pair (same vocab, different semantics)

""",

        "cola": """Generate semantically-aware perturbations that stay on-manifold (unlike random word dropout).

# CORE PRINCIPLES
- Use only masked vocabulary + allowed function words
- Generate natural, fluent text
- Follow task-specific strategies exactly
- Create effective contrastive pairs

# TASK RULES
Task: Grammar Checking (Acceptable/Unacceptable).
Signal: The structural error or the specific grammatical construct.
Noise: Valid words that do not affect grammaticality.
RULE: If the mask contains the error, the text MUST be Unacceptable.
RULE: If the mask removes the error (leaving only Noise), the text MUST be Acceptable (valid).

# PROTOCOL
1. Generate {n_samples} binary masks [{vocab_count} values] following task strategies
2. For each mask, create TWO infills using ONLY masked words:
   - neutral_infill: as defined in task rules
   - contrastive_infill: as defined in task rules
3. Use only masked_vocab + allowed function words (if specified)
""",

        "hatexplain": """Generate semantically-aware perturbations that stay on-manifold (unlike random word dropout).

# CORE PRINCIPLES
- Use only masked vocabulary + allowed function words
- Generate natural, fluent text
- Follow task-specific strategies exactly
- Create effective contrastive pairs

# TASK RULES
Task: Hate Speech Detection (Hatespeech/Offensive/Normal).
Signal: Slurs, attacks, or toxic terms.
Noise: Safe, benign words.
RULE: If the mask contains Signal words, the text MUST be Toxic/Offensive.
RULE: If the mask contains ONLY Noise words, the text MUST be Benign/Safe (clinical, no implied hate).

# PROTOCOL
1. Generate {n_samples} binary masks [{vocab_count} values] following task strategies
2. For each mask, create TWO infills using ONLY masked words:
   - neutral_infill: as defined in task rules
   - contrastive_infill: as defined in task rules
3. Use only masked_vocab + allowed function words (if specified)
""",
    },

    "v7": {
        "sst2": """Sentiment (Positive/Negative)

Masks ({n_samples} total - vary sizes):
- 40% random (2-6 words)
- 30% sentiment words (terrible, great, boring)
- 20% negation pairs (not good, never boring)
- 10% intensifiers (very disappointing, really great)

3 Infills:
primary_sample: Clear sentiment matching "{predicted_label}"
  Ex: [great, performance] → "The great performance"

contrastive_sample: Weak/mixed sentiment (NOT opposite)
  Ex: [great, performance] → "Great in parts"

minimal_sample: Fragment (2-5 words, no elaboration)
  Ex: [great, performance] → "great performance"

Allowed: a, an, the, is, are, was, were, and, or, but, in, at, to, of, for, with
Keep minimal under 5 words.""",

        "cola": """Grammar (Acceptable/Unacceptable)

Masks ({n_samples} total):
- 60% error only (1-2 words)
- 30% error + context (3-4 words)
- 10% valid words only

3 Infills:
primary_sample: Preserve label "{predicted_label}"
  If Unacceptable: keep error → "go"
  If Acceptable: keep correct → "goes"

contrastive_sample: Flip grammar
  If Unacceptable: fix → "goes"
  If Acceptable: break → "go"

minimal_sample: Just the word(s)
  Ex: [go] → "go"
  Ex: [are, cat] → "are cat"

Allowed: the, a, an, to, of, in, on, at, is, are
Single words OK.""",

        "hatexplain": """Hate Speech (Hatespeech/Offensive/Normal)

Masks ({n_samples} total):
- 50% toxic only (1-2 words)
- 30% toxic + context (3-5 words)
- 20% neutral only

3 Infills:
primary_sample: Direct with safety framing
  Toxic → "The [slur] as [attack]"
  Normal → "The group and people"

contrastive_sample: Academic/quoted framing
  Ex: "Text contains '[slur]' as attack"

minimal_sample: Minimal with quotes
  Ex: [slur] → "[slur]"
  Ex: [attack] → "term [attack]"

Allowed: the, a, an, as, in, of, about, term, phrase, text, contains
NEVER unframed toxic content."""
    },
    # for v6
    "v4":
        {
            "sst2": """**Task**: Sentiment (Positive/Negative)

**Mask Strategies** ({n_samples} total):
- 50% random: Random 2-5 words
- 30% sentiment_words: 1-3 sentiment adjectives/adverbs
- 20% negation_units: Negation + adjacent word

**Sample Generation**:
Create MINIMAL text using masked words:
- Just the words with minimal connectives
- NO elaborate phrasing or full sentences
- Fragments are better than fluency

Examples:
  mask=[great, performance] → "great performance"
  mask=[terrible, boring] → "terrible boring"
  mask=[not, good] → "not good"
  mask=[very, disappointing] → "very disappointing"
  mask=[movie, film, actor] → "movie film actor" or "the movie and film"

Allowed additions: a, an, the, is, are, and, or
Avoid: elaborate framing, descriptive context, complex structures""",

            "cola": """**Task**: Grammar (Acceptable/Unacceptable)

**Mask Strategies** ({n_samples} total):
- 60% error_only: Just the grammatical error (1-2 words)
- 30% error_minimal: Error + 1-2 essential words
- 10% valid_minimal: Minimal valid construction

**Sample Generation**:
Create MINIMAL text - often just 1-2 words:

Examples:
  mask=[go] from "He go store" → "go" or "He go"
  mask=[goes] from "He goes" → "goes"
  mask=[are, cat] from "The are cat" → "are cat" or "the are cat"
  mask=[is, cat] from "The cat is" → "is cat" or "cat is"

Single words are perfectly acceptable.
Minimal phrases test grammar directly.""",

            "hatexplain": """**Task**: Hate Speech Detection

**Mask Strategies** ({n_samples} total):
- 50% toxic_only: Slurs/toxic terms alone (1-2 words)
- 30% toxic_minimal: Toxic + 1-2 context words
- 20% neutral_only: Non-toxic descriptive words

**Sample Generation**:
Create MINIMAL text, with clinical distance for toxic content:

Examples:
  mask=[slur] → "[slur]" or "the term [slur]"
  mask=[slur, attack] → "[slur] [attack]" or "term [slur] attack"
  mask=[group, people] → "group people" or "the group"

For toxic terms: Use quotation marks or "the term X" for safety.
Keep it minimal - avoid elaborate academic framing."""
        },
    # For V5
    "v3": {"sst2": """**Task**: Sentiment (Positive/Negative)

**Features**:
- Critical: sentiment words (terrible, great), intensifiers (very), negation (not)
- Neutral: nouns (movie, film), function words

**Mask Strategies** ({n_samples} total):
- 50% random_exploration: Random 2-6 words (like vanilla LIME)
- 30% sentiment_core: 1-3 sentiment words only
- 20% negation_pairs: negation + adjacent word together

**SINGLE Infill Strategy**:
Generate ONLY neutral_infill (omit contrastive_infill):
- Use masked words naturally and fluently
- Maintain predicted label "{predicted_label}"
- Let model confidence vary based on which words present (don't force weakening)

Examples:
  mask=[great, performance] → "The great performance"
  mask=[boring, predictable] → "Boring and predictable"
  mask=[not, good] → "Not good"
  mask=[movie, film, plot] → "The movie and film have a plot"

**Constraints**:
- Allowed: a, an, the, is, are, was, were, in, at, to, of, for, with, and, or, have, has, had
- FORBIDDEN: some, but, though, however, yet, somewhat, partially, debatable, arguably
- No hedging or weakening phrases
- No contradictions or mixed signals
- Keep it simple and natural""",

           "cola": """**Task**: Grammar (Acceptable/Unacceptable)

**Features**:
- Critical: grammatical errors (subject-verb, word order, missing articles)
- Neutral: correctly formed words

**Mask Strategies** ({n_samples} total):
- 50% error_isolation: only the error (1-2 words)
- 30% error_context: error + 1-3 surrounding words  
- 20% valid_only: exclude error, include valid words

**DUAL Infill Strategy**:
neutral_infill: Preserve predicted label "{predicted_label}"
  If Unacceptable: keep error → "He go" or "go"
  If Acceptable: keep correct → "He goes" or "goes"

contrastive_infill: Flip grammaticality
  If Unacceptable: fix error → "goes" (not "go")
  If Acceptable: break grammar → "go" (not "goes")

**Constraints**:
- Allowed: the, a, an, to, of, in, on, at, is, are
- Single words or minimal phrases OK
- Create discrete correct ↔ incorrect flip""",

           "hatexplain": """**Task**: Hate Speech (Hatespeech/Offensive/Normal)

**Features**:
- Critical: slurs, dehumanizing terms, threats
- Neutral: descriptive identity terms, clinical language

**Mask Strategies** ({n_samples} total):
- 40% slur_isolation: toxic terms only (1-2 words)
- 35% slur_context: toxic terms + 2-4 surrounding words
- 25% neutral_only: exclude all toxic terms

**DUAL Infill Strategy**:
neutral_infill: Preserve label "{predicted_label}"
  Toxic → clinical framing: "The text contains [slur] as attack"
  Normal → descriptive: "Discussion of [group] and [people]"

contrastive_infill: Neutralize with academic distance
  Add quotes/framing: "Analysis of '[slur]' as attack language"
  Or: "The term [slur] appears in hate speech context"

**Constraints**:
- Allowed: the, a, an, as, in, of, about, when, used, appears, contains, term, phrase, text, analysis, language
- NEVER generate unframed hateful content
- Use quotation marks or "the term X" for distance
- Must be safe for human evaluators"""},
    "v2": {
        "sst2": """**TASK**: Sentiment Analysis (Positive/Negative)

**LABEL-CRITICAL FEATURES**:
- Sentiment adjectives: terrible, great, boring, excellent, awful, wonderful
- Sentiment adverbs: disappointingly, wonderfully, poorly, brilliantly
- Intensifiers: very, extremely, really, absolutely, quite
- Negations modifying sentiment: not good, never boring, no excitement

**NEUTRAL FEATURES**:
- Content nouns: movie, film, plot, character, actor, story
- Function words: the, a, it, this, that, is, was

**MASK GENERATION STRATEGIES**:
Generate {n_samples} masks using these strategies:

1. **sentiment_core** (40% of masks):
   - Include: 1-3 primary sentiment words
   - Exclude: neutral context words
   - Example: mask=[terrible, boring] from "This terrible boring movie"

2. **sentiment_plus_intensifier** (25% of masks):
   - Include: sentiment words + their intensifiers
   - Test if intensifiers amplify importance
   - Example: mask=[very, disappointing] from "very disappointing film"

3. **negation_scope** (20% of masks):
   - Include: negation + adjacent sentiment word as unit
   - Test compositional effects
   - Example: mask=[not, good] from "not good at all"

4. **random_exploration** (15% of masks):
   - Random selection of 2-5 words
   - Captures unexpected interactions like vanilla LIME

**INFILL TYPES**:

A. **neutral_infill** (Label-Preserving):
   - Goal: Maintain predicted label "{predicted_label}"
   - Strategy: Use masked words to express clear sentiment matching the label
   - Example for Positive prediction:
     * mask=[great, performance] → "The great performance stands out"
   - Example for Negative prediction:
     * mask=[terrible, boring] → "The terrible and boring execution fails"

B. **contrastive_infill** (Label-Weakening):
   - Type: "weak_infill" (NOT boundary-flipping)
   - Goal: Dilute sentiment while staying on-manifold
   - Strategy: Add ambiguity, mixed signals, or hedging using masked words
   - Example for Positive prediction:
     * mask=[great, performance] → "The performance is great in some aspects"
   - Example for Negative prediction:
     * mask=[terrible, boring] → "The film is boring at times, terrible in parts"
   
**CRITICAL CONSTRAINTS**:
- Allowed function words: a, an, the, is, are, was, were, be, in, on, at, to, of, for, with, some, but, and, or, though, while
- FORBIDDEN additions: debatable, arguably, questionable, supposedly, allegedly, or is it, not really, perhaps, maybe
- Both infills must use ONLY masked_vocab + allowed function words
- contrastive_infill should weaken (not flip) the sentiment
- Avoid contradictions like "great... but terrible" unless both words are in mask""",

        "cola": """**TASK**: Grammatical Acceptability (Acceptable/Unacceptable)

**LABEL-CRITICAL FEATURES**:
- Grammatical errors: subject-verb agreement violations, word order errors, missing articles
- Specific error patterns: "He go", "The are", "I eats", "She come"

**NEUTRAL FEATURES**:
- Grammatically valid words that don't affect the judgment
- Correctly conjugated verbs, proper word order

**MASK GENERATION STRATEGIES**:
Generate {n_samples} masks using these strategies:

1. **error_isolation** (50% of masks):
   - Include: ONLY the grammatical error (1-2 words)
   - Exclude: all other words
   - Example: mask=[go] from "He go to store"

2. **error_plus_context** (30% of masks):
   - Include: error + 1-3 surrounding words
   - Test if error needs context to be detected
   - Example: mask=[He, go] from "He go to store"

3. **valid_only** (20% of masks):
   - Exclude: the error entirely
   - Include: grammatically valid words
   - Example: mask=[He, to, store] (excluding "go")

**INFILL TYPES**:

A. **neutral_infill** (Label-Preserving):
   - Goal: Maintain predicted label "{predicted_label}"
   - If Unacceptable: Preserve the grammatical error naturally
     * mask=[go] from "He go" → "go" or "He go"
   - If Acceptable: Write grammatically correct phrase
     * mask=[He, goes] → "He goes"

B. **contrastive_infill** (Label-Flipping):
   - Type: "boundary_infill" (discrete flip for grammar)
   - Goal: Fix or introduce the grammatical error
   - If Unacceptable predicted: Fix the error
     * mask=[go] → "goes" (corrected)
   - If Acceptable predicted: Introduce error
     * mask=[goes] → "go" (made incorrect)

**CRITICAL CONSTRAINTS**:
- Allowed function words: the, a, an, to, of, in, on, at, is, are, was, were
- For error_isolation masks: infills can be single words or minimal phrases
- contrastive_infill should create discrete grammatical flip (correct ↔ incorrect)
- No need for complex sentences; minimal valid constructions are fine""",

        "hatexplain": """**TASK**: Hate Speech Detection (Hatespeech/Offensive/Normal)

**LABEL-CRITICAL FEATURES**:
- Slurs: racial, ethnic, religious, gender-based, sexual orientation slurs
- Attack language: dehumanizing terms, threats, violent rhetoric
- Toxic intensifiers: disgusting, subhuman, vermin, scum

**NEUTRAL FEATURES**:
- Identity terms used descriptively (not pejoratively)
- Neutral nouns, verbs, and discourse markers
- Academic or clinical language

**MASK GENERATION STRATEGIES**:
Generate {n_samples} masks using these strategies:

1. **slur_isolation** (40% of masks):
   - Include: ONLY slurs or toxic terms (1-2 words)
   - Exclude: all neutral context
   - Example: mask=[slur] from longer toxic text

2. **slur_plus_context** (35% of masks):
   - Include: toxic terms + 2-4 surrounding words
   - Test if context amplifies or moderates toxicity
   - Example: mask=[attack, slur, targets]

3. **neutral_only** (25% of masks):
   - Exclude: all toxic terms
   - Include: only neutral/descriptive words
   - Test if non-toxic words trigger false positives

**INFILL TYPES**:

A. **neutral_infill** (Label-Preserving):
   - Goal: Maintain predicted label "{predicted_label}"
   - If toxic label predicted: Use clinical/academic framing
     * mask=[slur, attack] → "The text contains [slur] as an [attack]"
     * Or quoted: "The phrase '[slur] [attack]' appears in the content"
   - If normal label predicted: Write neutral descriptive text
     * mask=[group, people] → "The group of people in question"

B. **contrastive_infill** (Label-Neutralizing):
   - Type: "clinical_infill" (reframe toxicity)
   - Goal: Neutralize or academically distance the toxic content
   - Strategy: Add quotation marks, clinical framing, or academic context
   - Example for toxic prediction:
     * mask=[slur, attack] → "An analysis of '[slur]' as [attack] language"
     * Or: "The term [slur] when used as [attack] indicates bias"
   - Example for normal prediction:
     * mask=[group, people] → "Discussion about [group] and [people]"

**CRITICAL CONSTRAINTS**:
- Allowed function words: the, a, an, as, in, of, about, when, used, appears, contains, indicates, analysis, term, phrase, text, language
- NEVER generate genuinely hateful content without clinical framing
- Use quotation marks, "the term X", or "the phrase X" to create distance
- contrastive_infill should reframe (not amplify) toxicity
- Both infills must be safe for human evaluators to read"""
    },
    "v1": {
        "sst2": """Task: Sentiment Analysis (Positive/Negative).
Signal: Adjectives, adverbs, and intensifiers (e.g., "terrible", "great").
Noise: Neutral nouns and function words (e.g., "movie", "film").
RULE: If the mask contains Signal words, the text MUST reflect the predicted label.
RULE: If the mask contains ONLY Noise words, the text MUST be Neutral (boring, factual, zero emotion).""",

        "cola": """Task: Grammar Checking (Acceptable/Unacceptable).
Signal: The structural error or the specific grammatical construct.
Noise: Valid words that do not affect grammaticality.
RULE: If the mask contains the error, the text MUST be Unacceptable.
RULE: If the mask removes the error (leaving only Noise), the text MUST be Acceptable (valid).""",

        "hatexplain": """Task: Hate Speech Detection (Hatespeech/Offensive/Normal).
Signal: Slurs, attacks, or toxic terms.
Noise: Safe, benign words.
RULE: If the mask contains Signal words, the text MUST be Toxic/Offensive.
RULE: If the mask contains ONLY Noise words, the text MUST be Benign/Safe (clinical, no implied hate).""",
    }
}
