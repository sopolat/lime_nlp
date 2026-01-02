# -*- coding: utf-8 -*-
"""Combine seed runs datasets

IG and SHAP have fixed values regardless of seed.


"""
import os
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
from lime_llm.log import get_logger

DATA_PATH = "../data/updated_baselines_test"
BASE_COLUMNS = ['dataset_name', 'idx', 'words', 'words_rationale']
BASE_METHODS = ['Partition SHAP', 'Integrated Gradient']
LIME_METHOD = ['LIME']

# Load environment variables from .env file
LOG = get_logger(__file__.replace(".py", ".log").replace("/lime_llm/", "/logs/"))
load_dotenv()


def main():
    LOG.warning("THE ORDER OF THE DATAFRAMES IS EXPECTED TO BE SAME WITHIN ALL FILES.")

    data_samples_filepaths_seed = {int(path.split("seed")[0].split("_")[-1]):
                                       os.path.join(DATA_PATH, path)
                                   for path in os.listdir(DATA_PATH)}

    data_samples_seed = {s: pd.read_csv(p) for s, p in data_samples_filepaths_seed.items()}

    first_seed = 42 #list(data_samples_filepaths_seed.keys())[0]

    base_df = data_samples_seed[first_seed][BASE_COLUMNS + BASE_METHODS]
    new_df = base_df.copy()

    for seed, df in tqdm(data_samples_seed.items(), total=len(data_samples_seed)):
        if len(df) != len(new_df):
            LOG.warning(f"seed {seed} has incorrect lengths: {len(df)} compared to {len(new_df)}")
            continue

        new_df[f"LIME_{seed}"] = df["LIME"].tolist()

    new_df_filpath = data_samples_filepaths_seed[first_seed].replace(f"{first_seed}seed", "lime_seeds")
    new_df.to_csv(new_df_filpath, index=False)
    LOG.info(f"Saved: {new_df_filpath}")


if __name__ == "__main__":
    main()
