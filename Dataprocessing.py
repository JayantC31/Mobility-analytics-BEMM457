import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

# load the data from the files

def load_and_clean_durations(folder_path):
    durations = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            if "Total duration (ms)" in df.columns:
                col = pd.to_numeric(df["Total duration (ms)"], errors="coerce")
                col = col.dropna()

                # Convert ms to minutes
                col = col / 60000

                durations.extend(col)

    durations = pd.Series(durations)

    # remove invalid durations
    durations = durations[durations > 0]

    # outliers
    lower = durations.quantile(0.01)
    upper = durations.quantile(0.99)
    durations = durations[(durations >= lower) & (durations <= upper)]

    # 3 hour cap for outlier
    durations = durations[durations <= 180]

    # drop duplicates
    durations = durations.drop_duplicates()

    return durations


# metric 1

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


# metric 2

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
    # weather dataset
    if not daily_counts:
        print("No weather datasets found.")
        return None

    df_weather = pd.DataFrame.from_dict(daily_counts, orient="index")

    df_weather["condition"] = df_weather["weather"].str.lower().apply(
        lambda x: "wet" if "rain" in x or "wet" in x else "dry"
    )
    # wet condition
    results = df_weather.groupby("condition")["count"].agg(
        Mean="mean",
        Standard_Deviation="std",
        Number_of_Days="count"
    )

    print("\nMetric 2: Weather-based Journey Frequency")
    print("-------------------------------------------")
    print(results)

    return results


# metric 3 

def calculate_metric_3(folder_path):

    peak_hours = list(range(7, 10)) + list(range(16, 19))
    offpeak_hours = [h for h in range(24) if h not in peak_hours]

    peak_counts = []
    offpeak_counts = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # start date
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

            # time and count
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


# main model

def build_main_model_count(folder_path, test_fraction=0.2, min_obs_per_mode=50,
                           max_points_resid_plot=5000):
    

    # load data
    all_dfs = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        required_cols = {"Date", "Time", "Count", "Weather"}
        if not required_cols.issubset(df.columns):
            continue

        # date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["Date"])

        # time to hour
        df["Time_parsed"] = pd.to_datetime(
            df["Time"].astype(str),
            errors="coerce",
            infer_datetime_format=True
        )
        df["hour"] = df["Time_parsed"].dt.hour
        df = df.dropna(subset=["hour"])

        # drop negatives
        df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
        df = df.dropna(subset=["Count"])
        df = df[df["Count"] >= 0]

        # wet and dry
        df["WetDay"] = df["Weather"].astype(str).str.lower().apply(
            lambda x: 1 if ("rain" in x or "wet" in x) else 0
        )

        # make weekend
        df["Weekend"] = df["Date"].dt.dayofweek.apply(lambda d: 1 if d >= 5 else 0)

        # peak hours
        peak_hours = {7, 8, 9, 16, 17, 18}
        df["PeakHour"] = df["hour"].apply(
            lambda h: 1 if int(h) in peak_hours else 0
        )

        # mode divider
        if "Mode" in df.columns:
            df["Mode"] = df["Mode"].astype(str)
        else:
            df["Mode"] = "All"

        subset = df[["Date", "hour", "Count", "WetDay", "Weekend", "PeakHour", "Mode"]].copy()
        subset.rename(
            columns={"hour": "HourOfDay", "Count": "JourneyCount"},
            inplace=True,
        )
        all_dfs.append(subset)

    if not all_dfs:
        print("No suitable aggregated weather/mode datasets found for the main model.")
        return None

    model_df = pd.concat(all_dfs, ignore_index=True)

    
    model_df["HourOfDay"] = model_df["HourOfDay"].astype(int)
    model_df["WetDay"] = model_df["WetDay"].astype(int)
    model_df["Weekend"] = model_df["Weekend"].astype(int)
    model_df["PeakHour"] = model_df["PeakHour"].astype(int)

    
    group_cols = ["Date", "HourOfDay", "WetDay", "Weekend", "PeakHour", "Mode"]

    print("Rows before aggregation:", len(model_df))
    model_df = (
        model_df
        .groupby(group_cols, as_index=False)
        .agg(JourneyCount=("JourneyCount", "sum"))
    )
    print("Rows after aggregation :", len(model_df))

    if model_df.empty:
        print("After aggregation, there are no rows left – check your filters / parsing.")
        return None

    # sort by time
    model_df = model_df.sort_values(["Date", "HourOfDay"]).reset_index(drop=True)

   # regression
    model_df["log_journeys"] = np.log1p(model_df["JourneyCount"])
    model_df["Hour_sq"] = model_df["HourOfDay"] ** 2

    mode_dummies = pd.get_dummies(model_df["Mode"], prefix="Mode")
    if mode_dummies.shape[1] > 1:
        # drop first as baseline
        mode_dummies = mode_dummies.iloc[:, 1:]
    model_df = pd.concat([model_df, mode_dummies], axis=1)

    base_features = ["HourOfDay", "Hour_sq", "PeakHour", "WetDay", "Weekend"]
    global_linear_features = base_features + list(mode_dummies.columns)

    # split based on time
    def time_based_split(df, date_col="Date", test_frac=0.2):
        """
        time
        """
        if df.empty:
            return df.copy(), df.copy(), None

        unique_dates = np.sort(df[date_col].dropna().unique())

        if len(unique_dates) == 0:
            return df.copy(), df.iloc[0:0].copy(), None
        elif len(unique_dates) == 1:
            return df.copy(), df.iloc[0:0].copy(), unique_dates[0]

        split_idx = int((1 - test_frac) * len(unique_dates))
        split_idx = max(1, min(split_idx, len(unique_dates) - 1))

        split_date = unique_dates[split_idx]
        train = df[df[date_col] <= split_date].copy()
        test = df[df[date_col] > split_date].copy()
        return train, test, split_date

    def fit_nb_and_evaluate(df_train, df_test, formula, label, make_residual_plot=True):
        
        
        if df_train.empty:
            print(f"\n{label}: no training data.")
            return None

        print(f"\n=== {label} ===")
        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

        glm_model = smf.glm(formula=formula, data=df_train,
                            family=sm.families.NegativeBinomial())
        result = glm_model.fit()

        
        null_model = smf.glm("JourneyCount ~ 1", data=df_train,
                             family=sm.families.NegativeBinomial()).fit()
        pseudo_r2 = 1 - result.deviance / null_model.deviance

        
        dispersion = result.pearson_chi2 / result.df_resid

        print("Pseudo R-squared (train):", round(pseudo_r2, 3))
        print("Dispersion (train):", round(dispersion, 3))

        # testing metrics
        rmse = None
        mae = None
        if df_test is not None and not df_test.empty:
            y_true = df_test["JourneyCount"].values
            y_pred = result.predict(df_test)
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            print("Test RMSE:", round(rmse, 3))
            print("Test MAE :", round(mae, 3))
        else:
            print("No test data for this model (all data in train).")

        residual_plot_data = None

        # residual plot 
        if make_residual_plot:
            fitted = result.fittedvalues
            residuals = df_train["JourneyCount"] - fitted

            plot_df = pd.DataFrame({"fitted": fitted, "resid": residuals})
            if len(plot_df) > max_points_resid_plot:
                plot_df = plot_df.sample(n=max_points_resid_plot, random_state=0)

            # csv 
            residual_plot_data = plot_df[["fitted", "resid"]].reset_index(drop=True)

            # plot
            jitter = np.random.normal(
                loc=0.0,
                scale=0.02 * plot_df["fitted"].std(),
                size=len(plot_df),
            )
            x_plot = plot_df["fitted"] + jitter

            plt.figure()
            plt.scatter(x_plot, plot_df["resid"], alpha=0.3, s=8)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Fitted values")
            plt.ylabel("Residuals (y - fitted)")
            plt.title(f"Residuals vs Fitted – {label} (train, sampled)")
            plt.tight_layout()
            plt.show()

        return {
            "model": result,
            "pseudo_r2_train": float(pseudo_r2),
            "dispersion_train": float(dispersion),
            "rmse_test": rmse,
            "mae_test": mae,
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
            "residual_plot_data": residual_plot_data,
        }

    def fit_linear_and_evaluate(df_train, df_test, feature_cols, label,
                                make_residual_plot=True):
        """
        linear
        """
        if df_train.empty:
            print(f"\n{label}: no training data.")
            return None

        print(f"\n=== {label} ===")
        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

        X_train = df_train[feature_cols].values
        y_train_log = df_train["log_journeys"].values

        model = LinearRegression()
        model.fit(X_train, y_train_log)

        r2_train = model.score(X_train, y_train_log)
        print("R-squared (train, log scale):", round(r2_train, 3))

        rmse = None
        mae = None
        if df_test is not None and not df_test.empty:
            X_test = df_test[feature_cols].values
            y_true = df_test["JourneyCount"].values
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)  # back to counts

            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            print("Test RMSE (counts):", round(rmse, 3))
            print("Test MAE  (counts):", round(mae, 3))
        else:
            print("No test data for this model (all data in train).")

        residual_plot_data = None

        # residual plot
        if make_residual_plot:
            X_train = df_train[feature_cols].values
            y_true_train = df_train["JourneyCount"].values
            y_pred_log_train = model.predict(X_train)
            y_pred_train = np.expm1(y_pred_log_train)

            plot_df = pd.DataFrame({
                "fitted": y_pred_train,
                "resid": y_true_train - y_pred_train
            })
            if len(plot_df) > max_points_resid_plot:
                plot_df = plot_df.sample(n=max_points_resid_plot, random_state=0)

            
            residual_plot_data = plot_df[["fitted", "resid"]].reset_index(drop=True)

            # plot
            jitter = np.random.normal(
                loc=0.0,
                scale=0.02 * plot_df["fitted"].std(),
                size=len(plot_df),
            )
            x_plot = plot_df["fitted"] + jitter

            plt.figure()
            plt.scatter(x_plot, plot_df["resid"], alpha=0.3, s=8)
            plt.axhline(0, linestyle="--")
            plt.xlabel("Fitted values (counts)")
            plt.ylabel("Residuals (y - fitted)")
            plt.title(f"Residuals vs Fitted – {label} (train, sampled)")
            plt.tight_layout()
            plt.show()

        return {
            "model": model,
            "r2_train_log": float(r2_train),
            "rmse_test": rmse,
            "mae_test": mae,
            "n_train": int(len(df_train)),
            "n_test": int(len(df_test)),
            "features": feature_cols,
            "residual_plot_data": residual_plot_data,
        }

    
    print("\n global:")

    global_nb_formula = "JourneyCount ~ C(HourOfDay) + PeakHour + WetDay + Weekend + C(Mode)"

    global_train, global_test, global_split_date = time_based_split(
        model_df, date_col="Date", test_frac=test_fraction
    )

    if global_split_date is not None:
        print(
            f"Global train: {global_train['Date'].min().date()} to {global_train['Date'].max().date()}"
        )
        if not global_test.empty:
            print(
                f"Global test : {global_test['Date'].min().date()} to {global_test['Date'].max().date()}"
            )
        else:
            print("Global test : (no test rows)")
    else:
        print("Global split date is None (no valid dates).")

    results = {
        "global": {},
        "by_mode": {},
        "data": model_df,
        "global_split_date": global_split_date,
        "global_nb_formula": global_nb_formula,
        "per_mode_nb_formula": "JourneyCount ~ C(HourOfDay) + PeakHour + WetDay + Weekend",
        "linear_features": {
            "global": global_linear_features,
            "base": base_features,
        },
    }

    # negative binomal
    results["global"]["negbin"] = fit_nb_and_evaluate(
        global_train,
        global_test,
        global_nb_formula,
        "Global Negative Binomial (all modes)",
        make_residual_plot=True,
    )

    # linear regression
    results["global"]["linear"] = fit_linear_and_evaluate(
        global_train,
        global_test,
        global_linear_features,
        "Global Linear Regression (log(JourneyCount + 1))",
        make_residual_plot=True,
    )

 
    print("\n per mode models:")

    per_mode_results = {}
    per_mode_nb_formula = "JourneyCount ~ C(HourOfDay) + PeakHour + WetDay + Weekend"

    for mode_value, df_mode in model_df.groupby("Mode"):
        print(f"\n----- Mode = {mode_value} -----")
        if len(df_mode) < min_obs_per_mode:
            print(f"Skipping mode '{mode_value}' (too few observations: {len(df_mode)})")
            continue

        df_mode = df_mode.sort_values(["Date", "HourOfDay"]).reset_index(drop=True)

        mode_train, mode_test, mode_split_date = time_based_split(
            df_mode, date_col="Date", test_frac=test_fraction
        )

        if mode_split_date is not None:
            print(
                f"Mode '{mode_value}' train: {mode_train['Date'].min().date()} to {mode_train['Date'].max().date()}"
            )
            if not mode_test.empty:
                print(
                    f"Mode '{mode_value}' test : {mode_test['Date'].min().date()} to {mode_test['Date'].max().date()}"
                )
            else:
                print(f"Mode '{mode_value}' test : (no test rows)")
        else:
            print(f"Mode '{mode_value}' split date is None (no valid dates).")

        mode_result = {
            "split_date": mode_split_date,
            "negbin": None,
            "linear": None,
        }

        # NegBin per mode
        mode_result["negbin"] = fit_nb_and_evaluate(
            mode_train,
            mode_test,
            per_mode_nb_formula,
            f"Negative Binomial (Mode = {mode_value})",
            make_residual_plot=True,
        )

        # linear
        mode_result["linear"] = fit_linear_and_evaluate(
            mode_train,
            mode_test,
            base_features,
            f"Linear Regression (log(JourneyCount + 1), Mode = {mode_value})",
            make_residual_plot=True,
        )

        per_mode_results[mode_value] = mode_result

    results["by_mode"] = per_mode_results


    try:
        nb_model = results["global"]["negbin"]["model"]
        lin_model = results["global"]["linear"]["model"]

        # NegBin predictions on all data
        model_df["pred_nb"] = nb_model.predict(model_df)

        # linear predictions on all data (convert back from log)
        X_all = model_df[global_linear_features].values
        pred_log_all = lin_model.predict(X_all)
        model_df["pred_lin"] = np.expm1(pred_log_all)

        hourly = model_df.groupby("HourOfDay").agg(
            actual_mean=("JourneyCount", "mean"),
            nb_mean=("pred_nb", "mean"),
            lin_mean=("pred_lin", "mean"),
        ).reset_index()

        plt.figure()
        plt.plot(hourly["HourOfDay"], hourly["actual_mean"], label="Actual mean/hour")
        plt.plot(hourly["HourOfDay"], hourly["nb_mean"], linestyle="--",
                 label="NegBin mean/hour")
        plt.plot(hourly["HourOfDay"], hourly["lin_mean"], linestyle=":",
                 label="Linear mean/hour")
        plt.xlabel("Hour of day")
        plt.ylabel("Journeys per hour (mean)")
        plt.title("Actual vs Predicted Journeys per Hour (Global Models)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        results["hourly_comparison"] = hourly

    except Exception as e:
        print("Skipping hourly comparison plot due to error:", e)

    # save for csv
    try:
        
        nb_resid = results["global"]["negbin"].get("residual_plot_data")
        if nb_resid is not None:
            nb_path = os.path.join(folder_path, "global_negbin_residuals.csv")
            nb_resid.to_csv(nb_path, index=False)
            print(f"Saved global NegBin residuals to: {nb_path}")

        
        lin_resid = results["global"]["linear"].get("residual_plot_data")
        if lin_resid is not None:
            lin_path = os.path.join(folder_path, "global_linear_residuals.csv")
            lin_resid.to_csv(lin_path, index=False)
            print(f"Saved global Linear residuals to: {lin_path}")

        
        if "hourly_comparison" in results:
            hourly_path = os.path.join(folder_path, "hourly_comparison.csv")
            results["hourly_comparison"].to_csv(hourly_path, index=False)
            print(f"Saved hourly comparison to: {hourly_path}")

    except Exception as e:
        print("Error saving CSV files:", e)

    return results


# USED TO RUN EVERYTHING

if __name__ == "__main__":
    folder_path = r"C:\Users\jcgam\OneDrive\Documents\Year4\TFL"

    #calculate_metric_1(folder_path)
    #calculate_metric_2(folder_path)
    #calculate_metric_3(folder_path)

    main_model_results = build_main_model_count(folder_path)
   

