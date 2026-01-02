#!/bin/bash

# Define the specific python executable
PYTHON_EXEC="/Users/georgemihaila/miniconda3/envs/lime_nlp_baselines/bin/python"

# Define the script to run
SCRIPT="lime_llm/script_lime_nlp_baselines_with_seed.py"

# Define the standard list of seeds
SEEDS=(0 1 123 1234 2023 2024 7 10 99 5 888 3407 1001 5555 13 21 256 512 1024 2048 1111 2222 3333 4321 9001 12345 54321 8080 314159)

# Get the total number of seeds
total_seeds=${#SEEDS[@]}
current_idx=0

echo "Starting batch processing of $total_seeds seeds..."

# Loop through each seed
for seed in "${SEEDS[@]}"
do
    # Increment counter
    ((current_idx++))

    # Calculate percentage
    percent=$(( 100 * current_idx / total_seeds ))

    # Define bar length (e.g., 50 characters)
    bar_len=50
    # Calculate how many hashtags to draw
    filled_len=$(( percent * bar_len / 100 ))

    # Create the bar string
    bar=$(printf "%-${bar_len}s" "#") # create a string of spaces
    bar="${bar// /#}"                # fill with hash temporarily
    # Cut the strings to correct lengths for filled vs empty
    filled_part=$(printf "%-${filled_len}s" "#")
    filled_part="${filled_part// /#}"
    empty_len=$(( bar_len - filled_len ))
    empty_part=$(printf "%-${empty_len}s" ".")

    # ------------------------------------------------
    # Print Progress Header
    # ------------------------------------------------
    echo ""
    echo "=================================================================="
    echo "Progress: [${filled_part}${empty_part}] ${percent}% ($current_idx / $total_seeds)"
    echo "Currently Running Seed: $seed"
    echo "=================================================================="

    # Run the script passing the seed as an argument
    $PYTHON_EXEC $SCRIPT --seed $seed

    echo ">> Finished seed $seed"
done

echo ""
echo "------------------------------------------------"
echo "All $total_seeds runs completed successfully."
echo "------------------------------------------------"