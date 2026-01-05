#!/usr/bin/env python3
"""
Arguments
"""

TASK_MODELS = {
        "sst2": "distilbert-base-uncased-finetuned-sst-2-english",
        "hatexplain": "gmihaila/bert-base-cased-hatexplain",
        "cola": "textattack/distilbert-base-uncased-CoLA"
    }
METHODS = ["LIME", "Partition SHAP", "Integrated Gradient", "LIME-LLM"]

LLM_SET = {
    "sonnet45": {
        "model": "claude-sonnet-4-5-20250929",
        "provider": "anthropic"  # "openai" or "anthropic"
    },
    "gpt5": {
        "model": "gpt-5-2025-08-07",
        "provider": "openai",
    },
    "gpt41": {
        "model": "gpt-4.1-2025-04-14",
        "provider": "openai",
    },
}