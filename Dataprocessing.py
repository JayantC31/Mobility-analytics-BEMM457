import os
import pandas as pd
import numpy as np

folder_path = r"C:\Users\jcgam\OneDrive\Documents\Year4\TFL"

durations = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Check column
        if "Total duration (ms)" in df.columns:
            col = pd.to_numeric(df["Total duration (ms)"], errors="coerce")

            # Drop missing values
            col = col.dropna()

            # Convert ms → minutes
            col = col / 60000

            durations.extend(col)

# Convert to pandas Series
durations = pd.Series(durations)

# ----------------------------------------------------------
# CLEANING STEPS (for Task 2)
# ----------------------------------------------------------

# 1. Remove non-positive durations (0 or negative)
durations = durations[durations > 0]

# 2. Remove unrealistic tiny trips (< 1st percentile)
lower_limit = durations.quantile(0.01)
durations = durations[durations >= lower_limit]

# 3. Remove extreme long trips (> 99th percentile)
upper_limit = durations.quantile(0.99)
durations_clean = durations[durations <= upper_limit]

# 4. Optional cap: remove anything > 3 hours (180 minutes)
durations_clean = durations_clean[durations_clean <= 180]

# 5. Remove duplicates
durations_clean = durations_clean.drop_duplicates()

# ----------------------------------------------------------
# FINAL CLEANED STATISTICS
# ----------------------------------------------------------

print("Cleaned Journey Duration Statistics (Minutes)")
print("--------------------------------------------")
print("Mean:", durations_clean.mean())
print("Median:", durations_clean.median())
print("Standard Deviation:", durations_clean.std())


