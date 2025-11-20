import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CLEANING PIPELINE (runs once and shared by all metrics)
# ---------------------------------------------------------

def load_and_clean_durations(folder_path):
    durations = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if "Total duration (ms)" in df.columns:
                col = pd.to_numeric(df["Total duration (ms)"], errors="coerce")
                col = col.dropna()

                # Convert ms → minutes
                col = col / 60000

                durations.extend(col)

    durations = pd.Series(durations)

    # Remove invalid durations
    durations = durations[durations > 0]

    # Remove extreme short + long
    lower = durations.quantile(0.01)
    upper = durations.quantile(0.99)
    durations = durations[(durations >= lower) & (durations <= upper)]

    # Cap at 3 hours
    durations = durations[durations <= 180]

    # Drop duplicates
    durations = durations.drop_duplicates()

    return durations


# ---------------------------------------------------------
# METRIC 1 – Journey Duration Statistics
# ---------------------------------------------------------

def calculate_metric_1(folder_path):
    durations_clean = load_and_clean_durations(folder_path)

    print("\nMetric 1: Journey Duration Statistics (minutes)")
    print("------------------------------------------------")
    print("Mean:", round(durations_clean.mean(), 2))
    print("Median:", round(durations_clean.median(), 2))
    print("Standard Deviation:", round(durations_clean.std(), 2))

    return {
        "mean": durations_clean.mean(),
        "median": durations_clean.median(),
        "std": durations_clean.std()
    }


# ---------------------------------------------------------
# METRIC 2 – Weather-Based Journey Frequency
# ---------------------------------------------------------

def calculate_metric_2(folder_path):
    daily_counts = {}

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if {"Weather", "Count", "Date"}.issubset(df.columns):
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
                df = df.dropna(subset=["Date", "Weather", "Count"])

                grouped = df.groupby("Date").agg({
                    "Weather": "first",
                    "Count": "sum"
                }).reset_index()

                for _, row in grouped.iterrows():
                    date = row["Date"].date()
                    daily_counts[date] = {
                        "weather": row["Weather"],
                        "count": row["Count"]
                    }

    if not daily_counts:
        print("No weather datasets found.")
        return None

    df_weather = pd.DataFrame.from_dict(daily_counts, orient="index")

    df_weather["condition"] = df_weather["weather"].str.lower().apply(
        lambda x: "wet" if "rain" in x or "wet" in x else "dry"
    )

    results = df_weather.groupby("condition")["count"].agg(
        Mean="mean",
        Standard_Deviation="std",
        Number_of_Days="count"
    )

    print("\nMetric 2: Weather-based Journey Frequency")
    print("-------------------------------------------")
    print(results)

    return results


# ---------------------------------------------------------
# METRIC 3 – Peak vs Off-Peak Hourly Frequency
# ---------------------------------------------------------

def calculate_metric_3(folder_path):

    peak_hours = list(range(7, 10)) + list(range(16, 19))
    offpeak_hours = [h for h in range(24) if h not in peak_hours]

    peak_counts = []
    offpeak_counts = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # CASE A: Trip-level dataset with "Start date"
            if "Start date" in df.columns:
                df["Start date"] = pd.to_datetime(
                    df["Start date"], 
                    errors="coerce", 
                    format=None   # prevents warnings
                )

                df = df.dropna(subset=["Start date"])
                df["hour"] = df["Start date"].dt.hour

                hours = df["hour"].value_counts()

                for hour, count in hours.items():
                    if hour in peak_hours:
                        peak_counts.append(count)
                    else:
                        offpeak_counts.append(count)

            # CASE B: Aggregated dataset with "Time" + "Count"
            elif {"Time", "Count"}.issubset(df.columns):
                df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)

                df = df[df["Count"] > 0]

                df["hour"] = df["Time"].astype(str).str.strip().str[:2]
                df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

                df = df.dropna(subset=["hour"])

                for _, row in df.iterrows():
                    hour = int(row["hour"])
                    c = row["Count"]
                    if hour in peak_hours:
                        peak_counts.append(c)
                    else:
                        offpeak_counts.append(c)

    peak_counts = pd.Series(peak_counts)
    offpeak_counts = pd.Series(offpeak_counts)

    print("\nMetric 3: Peak vs Off-Peak Journey Frequency")
    print("---------------------------------------------")
    print("Peak Mean Per Hour:", peak_counts.mean())
    print("Peak Std:", peak_counts.std())
    print("Peak Total:", peak_counts.sum())
    print()
    print("Off-Peak Mean Per Hour:", offpeak_counts.mean())
    print("Off-Peak Std:", offpeak_counts.std())
    print("Off-Peak Total:", offpeak_counts.sum())

    return {
        "peak_mean": peak_counts.mean(),
        "peak_std": peak_counts.std(),
        "peak_total": peak_counts.sum(),
        "off_peak_mean": offpeak_counts.mean(),
        "off_peak_std": offpeak_counts.std(),
        "off_peak_total": offpeak_counts.sum()
    }


# ---------------------------------------------------------
# RUN EVERYTHING
# ---------------------------------------------------------

if __name__ == "__main__":
    folder_path = r"C:\Users\jcgam\OneDrive\Documents\Year4\TFL"

    #calculate_metric_1(folder_path)
    #calculate_metric_2(folder_path)
    #calculate_metric_3(folder_path)
