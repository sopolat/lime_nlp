#!/usr/bin/env python3
"""
Arguments
"""

import warnings

from .prompts import SYSTEM_PROMPT_VERSIONS, USER_PROMPT_VERSIONS, DATASET_DESCRIPTION

warnings.filterwarnings("ignore")


class BaseArgs(object):
    """
    LLM-Enhanced LIME Evaluation Pipeline arguments.
    """

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    # SEED = 0  # SEEDS = [42, 0, 1, 123, 1234, 2023, 2024, 7, 10, 99]
    # TEST_MODE = False  # Set to True to run 1 example per dataset for testing
    # SYSTEM_PROMPT_USE_VERSION = "v9"
    # USER_PROMPT_USE_VERSION = "v9"
    # DATASET_DESCRIPTION_VERSION = "v9"
    # LLM_NAME = "sonnet45"  # sonnet45 gpt41
    # SENTENCE_TRANSFORMER_MODEL = "all-mpnet-base-v2"
    # TEMPERATURE = 0.0
    # DATA_SAMPLES = 30  # 30 60 150
    # PERTURBATION_TYPE_SAMPLES = 10  # 30 TOTAL samples is ideal
    # WANDB_ENABLED = True
    # WANDB_PROJECT = "lime-nlp"
    # WANDB_ENTITY = os.getenv("WANDB_ENTITY")  # optional
    # WANDB_MODE = os.getenv("WANDB_MODE", "online")  # "online" | "offline" | "disabled"
    #
    # LLM_PROVIDER = LLM_SET[LLM_NAME]["provider"]
    # LLM_MODEL = LLM_SET[LLM_NAME]["model"]
    # DATA_PATH = f"data/original/xai_combined_df_{DATA_SAMPLES}_examples.csv"
    # OUTPUT_DIR = f"outputs/{'test' if TEST_MODE else 'run'}_{LLM_NAME}_{os.path.basename(DATA_PATH).removesuffix(".csv")}_sys{SYSTEM_PROMPT_USE_VERSION}_usr{USER_PROMPT_USE_VERSION}_datadesc{DATASET_DESCRIPTION_VERSION}"
    # LOG_FILE = f"{OUTPUT_DIR}/llm_calls.jsonl"
    # Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """
        Combine class and instance attributes
        """
        dict_arguments = {**self.__class__.__dict__}
        dict_arguments = {k: str(v) for k, v in dict_arguments.items()}
        dict_arguments.update({
            "system_prompt_version": SYSTEM_PROMPT_VERSIONS[self.SYSTEM_PROMPT_USE_VERSION],
            "user_prompt_version": USER_PROMPT_VERSIONS[self.USER_PROMPT_USE_VERSION],
            "dataset_description_version": DATASET_DESCRIPTION[self.DATASET_DESCRIPTION_VERSION], })
        return dict_arguments
