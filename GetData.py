import os
import pandas as pd

import os
import pandas as pd

folder_path = r"C:\Users\jcgam\OneDrive\Documents\Year4\TFL"

columns_to_drop = [
    "Wave", "SiteID", "Day", "Round", "Direction",
    "Number", "Start station number", "End station number",
    "Bike number", "Bike model"
]

print("Looking in folder:", folder_path)
print()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Detect CSV
    if filename.lower().endswith(".csv"):
        print(f"--- Processing CSV: {filename} ---")
        df = pd.read_csv(file_path)

    # Detect Excel
    elif filename.lower().endswith((".xlsx", ".xls")):
        print(f"--- Processing Excel: {filename} ---")
        df = pd.read_excel(file_path)

    else:
        continue  # skip non-data files

    # Normalize columns (important!)
    df.columns = df.columns.str.strip()  # remove spaces
    df.columns = df.columns.str.replace("\ufeff", "")  # remove BOM if present

    # Identify columns to drop
    cols_present = [col for col in columns_to_drop if col in df.columns]

    print("Columns in file:", list(df.columns))
    print("Dropping:", cols_present)

    # Drop if present
    if cols_present:
        df = df.drop(columns=cols_present)

        # Save back in correct format
        if filename.lower().endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)

        print("✔ File updated")
    else:
        print("⚠ No matching columns to drop")

    print()

print("Done.")





