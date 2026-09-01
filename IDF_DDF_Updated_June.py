# ============================================================
# COMPLETE IDF ANALYSIS (MULTI-DISTRIBUTION)
# USING ANNUAL MAXIMUM RAINFALL DATA
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, lognorm, gumbel_r, genextreme, skew, kstest, chi2, anderson

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_PATH = r"E:\BEST Pak\Meteorological Data\Frequency_Analysis"

INPUT_FILE = os.path.join(
    BASE_PATH,
    "Annual_Maximum_Rainfall.xlsx"
)

OUTPUT_FILE = os.path.join(
    BASE_PATH,
    "IDF_All_Distributions.xlsx"
)

PLOT_FOLDER = os.path.join(
    BASE_PATH,
    "IDF_Plots"
)

os.makedirs(PLOT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# READ ANNUAL MAXIMUM RAINFALL DATA
# ------------------------------------------------------------
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()

required_cols = ["Year", "Rainfall_mm"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Required column '{col}' not found in Excel file."
        )

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Rainfall_mm"] = pd.to_numeric(
    df["Rainfall_mm"],
    errors="coerce"
)

df = df.dropna(
    subset=["Year", "Rainfall_mm"]
)

years = df["Year"].astype(int)
data = df["Rainfall_mm"].values

print("="*60)
print("ANNUAL MAXIMUM RAINFALL DATA SUMMARY")
print("="*60)
print(f"✔ Data loaded successfully")
print(f"   Number of years: {len(data)}")
print(f"   Period: {years.min()} - {years.max()}")
print(f"   Data range: {data.min():.1f} to {data.max():.1f} mm")
print(f"   Mean annual maximum: {data.mean():.1f} mm")
print(f"   Standard deviation: {data.std():.1f} mm")

# ------------------------------------------------------------
# RETURN PERIODS
# ------------------------------------------------------------
T = np.array([2, 3, 4, 5, 10, 20, 50, 100, 500, 1000])
P = 1 - 1 / T

print(f"\nReturn periods considered (years): {T}")

# ------------------------------------------------------------
# DISTRIBUTION FITTING (24-HOUR ANNUAL MAXIMA)
# ------------------------------------------------------------
dist_results = {}
dist_params = {}
dist_objects = {}

print("\n" + "="*60)
print("DISTRIBUTION FITTING")
print("="*60)

# 1. NORMAL DISTRIBUTION
mu_norm, sigma_norm = norm.fit(data)
dist_objects["Normal"] = norm(mu_norm, sigma_norm)
dist_results["Normal"] = norm.ppf(P, mu_norm, sigma_norm)
dist_params["Normal"] = {
    "Distribution": "Normal",
    "Parameter_1": f"μ = {mu_norm:.3f}",
    "Parameter_2": f"σ = {sigma_norm:.3f}",
    "Parameter_3": "",
    "Parameter_4": ""
}
print(f"✓ Normal: μ={mu_norm:.3f}, σ={sigma_norm:.3f}")

# 2. LOGNORMAL DISTRIBUTION
shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0)
dist_objects["LogNormal"] = lognorm(shape_ln, loc_ln, scale_ln)
ln_values = lognorm.ppf(P, shape_ln, loc_ln, scale_ln)
ln_values[ln_values < 0] = np.nan
dist_results["LogNormal"] = ln_values
dist_params["LogNormal"] = {
    "Distribution": "LogNormal",
    "Parameter_1": f"shape = {shape_ln:.3f}",
    "Parameter_2": f"loc = {loc_ln:.3f}",
    "Parameter_3": f"scale = {scale_ln:.3f}",
    "Parameter_4": ""
}
print(f"✓ LogNormal: shape={shape_ln:.3f}, scale={scale_ln:.3f}")

# 3. GUMBEL DISTRIBUTION
loc_gumb, scale_gumb = gumbel_r.fit(data)
dist_objects["Gumbel"] = gumbel_r(loc_gumb, scale_gumb)
dist_results["Gumbel"] = gumbel_r.ppf(P, loc_gumb, scale_gumb)
dist_params["Gumbel"] = {
    "Distribution": "Gumbel",
    "Parameter_1": f"loc = {loc_gumb:.3f}",
    "Parameter_2": f"scale = {scale_gumb:.3f}",
    "Parameter_3": "",
    "Parameter_4": ""
}
print(f"✓ Gumbel: loc={loc_gumb:.3f}, scale={scale_gumb:.3f}")

# 4. GEV DISTRIBUTION (Generalized Extreme Value)
c_gev, loc_gev, scale_gev = genextreme.fit(data)
dist_objects["GEV"] = genextreme(c_gev, loc_gev, scale_gev)
dist_results["GEV"] = genextreme.ppf(P, c_gev, loc_gev, scale_gev)
dist_params["GEV"] = {
    "Distribution": "GEV",
    "Parameter_1": f"c = {c_gev:.3f}",
    "Parameter_2": f"loc = {loc_gev:.3f}",
    "Parameter_3": f"scale = {scale_gev:.3f}",
    "Parameter_4": ""
}
print(f"✓ GEV: c={c_gev:.3f}, loc={loc_gev:.3f}, scale={scale_gev:.3f}")

# 5. LP3 DISTRIBUTION (Log-Pearson Type III)
log_data = np.log10(data)
mean_lp3 = log_data.mean()
std_lp3 = log_data.std(ddof=1)
cs_lp3 = skew(log_data)
z = norm.ppf(P)
lp3_log = mean_lp3 + z * std_lp3 * (1 + (cs_lp3 * z / 6))
dist_results["LP3"] = 10 ** lp3_log
dist_objects["LP3"] = norm(mean_lp3, std_lp3)
dist_params["LP3"] = {
    "Distribution": "LP3",
    "Parameter_1": f"μ_log = {mean_lp3:.3f}",
    "Parameter_2": f"σ_log = {std_lp3:.3f}",
    "Parameter_3": f"γ_log = {cs_lp3:.3f}",
    "Parameter_4": ""
}
print(f"✓ LP3: μ_log={mean_lp3:.3f}, σ_log={std_lp3:.3f}, γ_log={cs_lp3:.3f}")

# Convert parameters to DataFrame
params_list = []
for dist_name, params in dist_params.items():
    params_list.append(params)

params_df = pd.DataFrame(params_list)
params_df = params_df[["Distribution", "Parameter_1",
                       "Parameter_2", "Parameter_3", "Parameter_4"]]

# ------------------------------------------------------------
# CHI-SQUARE GOODNESS OF FIT TEST
# ------------------------------------------------------------


def chi2_test(data, dist_obj, n_bins=8):
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(data, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    observed, _ = np.histogram(data, bins=bin_edges, density=False)

    expected = []
    for i in range(len(observed)):
        if i == 0:
            prob = dist_obj.cdf(bin_edges[i+1])
        elif i == len(observed) - 1:
            prob = 1 - dist_obj.cdf(bin_edges[i])
        else:
            prob = dist_obj.cdf(bin_edges[i+1]) - dist_obj.cdf(bin_edges[i])
        expected.append(prob * len(data))

    observed_final = []
    expected_final = []
    pooled_obs = 0
    pooled_exp = 0

    for i in range(len(observed)):
        if expected[i] < 5:
            pooled_obs += observed[i]
            pooled_exp += expected[i]
        else:
            if pooled_obs > 0:
                observed_final.append(pooled_obs)
                expected_final.append(pooled_exp)
                pooled_obs = 0
                pooled_exp = 0
            observed_final.append(observed[i])
            expected_final.append(expected[i])

    if pooled_obs > 0:
        observed_final.append(pooled_obs)
        expected_final.append(pooled_exp)

    chi2_stat = np.sum(((np.array(observed_final) -
                       np.array(expected_final))**2) / np.array(expected_final))
    n_params = len(dist_obj.args) - 1 if hasattr(dist_obj, 'args') else 2
    dof = max(1, len(observed_final) - 1 - n_params)
    p_value = 1 - chi2.cdf(chi2_stat, dof)

    return chi2_stat, p_value, dof

# ------------------------------------------------------------
# ANDERSON-DARLING TEST
# ------------------------------------------------------------


def ad_test(data, dist_obj, dist_name):
    if dist_name == 'Normal':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = anderson(data, dist='norm')
        critical_95 = result.critical_values[2]
        stat = result.statistic
        decision = "Do not reject" if stat < critical_95 else "Reject"

    elif dist_name == 'LogNormal':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = anderson(np.log(data), dist='norm')
        critical_95 = result.critical_values[2]
        stat = result.statistic
        decision = "Do not reject" if stat < critical_95 else "Reject"

    else:
        data_sorted = np.sort(data)
        n = len(data)
        cdf_vals = dist_obj.cdf(data_sorted)
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        S = np.sum((2 * np.arange(1, n+1) - 1) *
                   (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
        stat = -n - S/n
        critical_95 = 2.492
        decision = "Do not reject" if stat < critical_95 else "Reject"

    return stat, critical_95, decision


# ------------------------------------------------------------
# COMPREHENSIVE GOODNESS OF FIT TABLE
# ------------------------------------------------------------
print("\n" + "="*60)
print("PERFORMING GOODNESS OF FIT TESTS")
print("="*60)

gof_results = []

distributions = [
    ("Normal", dist_objects["Normal"], "Normal", data),
    ("LogNormal", dist_objects["LogNormal"], "LogNormal", data),
    ("Gumbel", dist_objects["Gumbel"], "Gumbel", data),
    ("GEV", dist_objects["GEV"], "GEV", data),
    ("LP3", dist_objects["LP3"], "Normal", log_data)
]

for name, dist_obj, test_name, test_data in distributions:
    if name == "LP3":
        ks_stat, ks_pvalue = kstest(
            test_data, "norm", args=(dist_obj.mean(), dist_obj.std()))
    elif name == "LogNormal":
        ks_stat, ks_pvalue = kstest(
            test_data, "lognorm", args=(shape_ln, 0, scale_ln))
    elif name == "Gumbel":
        ks_stat, ks_pvalue = kstest(
            test_data, "gumbel_r", args=(loc_gumb, scale_gumb))
    elif name == "GEV":
        ks_stat, ks_pvalue = kstest(
            test_data, "genextreme", args=(c_gev, loc_gev, scale_gev))
    else:
        ks_stat, ks_pvalue = kstest(
            test_data, "norm", args=(mu_norm, sigma_norm))

    ks_decision = "Do not reject" if ks_pvalue > 0.05 else "Reject"
    chi2_stat, chi2_pvalue, chi2_dof = chi2_test(test_data, dist_obj, n_bins=8)
    chi2_decision = "Do not reject" if chi2_pvalue > 0.05 else "Reject"
    ad_stat, ad_critical, ad_decision = ad_test(test_data, dist_obj, test_name)

    if name == "Normal":
        params_str = f"μ={mu_norm:.3f}, σ={sigma_norm:.3f}"
    elif name == "LogNormal":
        params_str = f"shape={shape_ln:.3f}, scale={scale_ln:.3f}"
    elif name == "Gumbel":
        params_str = f"loc={loc_gumb:.3f}, scale={scale_gumb:.3f}"
    elif name == "GEV":
        params_str = f"c={c_gev:.3f}, loc={loc_gev:.3f}, scale={scale_gev:.3f}"
    else:
        params_str = f"μ_log={mean_lp3:.3f}, σ_log={std_lp3:.3f}, γ_log={cs_lp3:.3f}"

    gof_results.append([
        name, params_str,
        ks_stat, ks_pvalue, ks_decision,
        chi2_stat, chi2_pvalue, chi2_dof, chi2_decision,
        ad_stat, ad_critical, ad_decision
    ])

    print(f"✓ {name}: KS p={ks_pvalue:.4f}, χ² p={chi2_pvalue:.4f}, AD={ad_stat:.4f}")

gof_df = pd.DataFrame(
    gof_results,
    columns=[
        "Distribution", "Parameters",
        "KS_Statistic", "KS_PValue", "KS_Decision",
        "Chi2_Statistic", "Chi2_PValue", "Chi2_DOF", "Chi2_Decision",
        "AD_Statistic", "AD_Critical_95", "AD_Decision"
    ]
)

# ------------------------------------------------------------
# CALCULATE COMPREHENSIVE SCORES FOR EACH DISTRIBUTION
# ------------------------------------------------------------
print("\n" + "="*60)
print("CALCULATING DISTRIBUTION SCORES")
print("="*60)

score_results = []

for idx, row in gof_df.iterrows():
    dist_name = row['Distribution']

    # Score 1: Combined test rejection score (0-3)
    rejection_score = 0
    if row['KS_Decision'] == "Reject":
        rejection_score += 1
    if row['Chi2_Decision'] == "Reject":
        rejection_score += 1
    if row['AD_Decision'] == "Reject":
        rejection_score += 1

    # Score 2: KS statistic (lower is better) - normalized to 0-100
    ks_score = row['KS_Statistic'] * 100

    # Score 3: KS p-value (higher is better) - normalized to 0-100
    ks_pvalue_score = row['KS_PValue'] * 100

    # Score 4: Chi-square statistic (lower is better) - normalized
    chi2_score = min(row['Chi2_Statistic'] * 10, 100)

    # Score 5: Chi-square p-value (higher is better)
    chi2_pvalue_score = row['Chi2_PValue'] * 100

    # Score 6: AD statistic (lower is better) - normalized
    ad_score = min(row['AD_Statistic'] * 20, 100)

    # Score 7: Combined rank (composite score)
    # Lower rejection score is better, lower KS/AD/Chi2 is better, higher p-values are better
    composite_score = (
        (rejection_score * 10) +  # Weight rejection more heavily
        (row['KS_Statistic'] * 50) +
        ((1 - row['KS_PValue']) * 20) +
        (row['Chi2_Statistic'] * 5) +
        ((1 - row['Chi2_PValue']) * 15) +
        (row['AD_Statistic'] * 20)
    )

    # Overall ranking (1 = best)
    overall_rank = composite_score

    score_results.append({
        "Distribution": dist_name,
        "Rejection_Score_(0-3)": rejection_score,
        "KS_Statistic": row['KS_Statistic'],
        "KS_PValue": row['KS_PValue'],
        "KS_Score_(lower_better)": f"{ks_score:.2f}",
        "KS_PValue_Score_(higher_better)": f"{ks_pvalue_score:.2f}",
        "Chi2_Statistic": row['Chi2_Statistic'],
        "Chi2_PValue": row['Chi2_PValue'],
        "Chi2_Score_(lower_better)": f"{chi2_score:.2f}",
        "Chi2_PValue_Score": f"{chi2_pvalue_score:.2f}",
        "AD_Statistic": row['AD_Statistic'],
        "AD_Score_(lower_better)": f"{ad_score:.2f}",
        "Composite_Score": f"{composite_score:.2f}",
        "Overall_Rank": ""
    })

# Sort by composite score and add ranks
score_df = pd.DataFrame(score_results)
score_df = score_df.sort_values('Composite_Score', ascending=True)
score_df['Overall_Rank'] = range(1, len(score_df) + 1)

# Add ranking description
rank_descriptions = {
    1: "⭐ BEST - Highly Recommended",
    2: "✓ GOOD - Recommended",
    3: "✓ ACCEPTABLE - Can be used",
    4: "⚠️ FAIR - Use with caution",
    5: "❌ POOR - Not recommended"
}
score_df['Recommendation'] = score_df['Overall_Rank'].map(
    lambda x: rank_descriptions.get(x, ""))

# Reset index
score_df = score_df.reset_index(drop=True)

# Create a summary scorecard
summary_card = pd.DataFrame({
    "Metric": [
        "Best Distribution",
        "Lowest KS Statistic",
        "Highest KS P-value",
        "Lowest Chi-square Statistic",
        "Highest Chi-square P-value",
        "Lowest AD Statistic",
        "Lowest Rejection Score"
    ],
    "Distribution": [
        score_df.iloc[0]['Distribution'],
        gof_df.loc[gof_df['KS_Statistic'].idxmin(), 'Distribution'],
        gof_df.loc[gof_df['KS_PValue'].idxmax(), 'Distribution'],
        gof_df.loc[gof_df['Chi2_Statistic'].idxmin(), 'Distribution'],
        gof_df.loc[gof_df['Chi2_PValue'].idxmax(), 'Distribution'],
        gof_df.loc[gof_df['AD_Statistic'].idxmin(), 'Distribution'],
        gof_df.loc[gof_df['KS_Decision'].apply(
            lambda x: 0 if x == "Do not reject" else 1).idxmin(), 'Distribution']
    ],
    "Value": [
        f"Composite Score: {score_df.iloc[0]['Composite_Score']}",
        f"{gof_df['KS_Statistic'].min():.5f}",
        f"{gof_df['KS_PValue'].max():.4f}",
        f"{gof_df['Chi2_Statistic'].min():.2f}",
        f"{gof_df['Chi2_PValue'].max():.4f}",
        f"{gof_df['AD_Statistic'].min():.4f}",
        "0 (All tests passed)"
    ]
})

# Display results
print("\n" + "="*80)
print("DISTRIBUTION COMPREHENSIVE SCORES")
print("="*80)
print(score_df.to_string(index=False))
print("\n" + "="*80)
print("SUMMARY SCORECARD")
print("="*80)
print(summary_card.to_string(index=False))

# ------------------------------------------------------------
# DURATIONS FOR IDF/DDF ANALYSIS
# ------------------------------------------------------------
durations_min = np.array(
    [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440, 2880, 4320])
durations_hr = durations_min / 60
duration_labels = ["5min", "10min", "15min", "30min", "1hr",
                   "2hrs", "3hrs", "6hrs", "12hrs", "24hrs", "48hrs", "72hrs"]

# ------------------------------------------------------------
# IDF GENERATION FUNCTION
# ------------------------------------------------------------


def generate_idf(depth_24):
    idf = pd.DataFrame(index=durations_min)
    for i, t in enumerate(T):
        depth_d = depth_24[i] * (durations_hr / 24) ** 0.33
        intensity = depth_d / durations_hr
        intensity[intensity < 0] = np.nan
        idf[f"{t}-yr"] = intensity
    idf.index.name = "Duration (min)"
    return idf


def generate_ddf(idf_table):
    ddf = idf_table.copy()
    for col in ddf.columns:
        ddf[col] = ddf[col] * durations_hr
    ddf.index.name = "Duration (min)"
    return ddf


# ------------------------------------------------------------
# GENERATE IDF AND DDF TABLES
# ------------------------------------------------------------
print("\n" + "="*60)
print("GENERATING IDF AND DDF TABLES")
print("="*60)

idf_tables = {}
ddf_tables = {}

for name, values in dist_results.items():
    idf_tables[name] = generate_idf(values)
    ddf_tables[name] = generate_ddf(idf_tables[name])
    print(f"✓ {name}: IDF/DDF tables generated")

# ------------------------------------------------------------
# SAVE RESULTS TO EXCEL
# ------------------------------------------------------------
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

try:
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except PermissionError:
            print(f"⚠ File {OUTPUT_FILE} is currently open.")
            alternative_file = OUTPUT_FILE.replace(".xlsx", f"_new.xlsx")
            print(f"   Saving to alternative file: {alternative_file}")
            OUTPUT_FILE = alternative_file

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Annual maxima data
        pd.DataFrame({
            "Year": years,
            "Rainfall_mm": data
        }).to_excel(writer, sheet_name="Annual_Maxima", index=False)

        # Goodness of fit results
        gof_df.to_excel(writer, sheet_name="Goodness_of_Fit", index=False)

        # Distribution parameters
        params_df.to_excel(
            writer, sheet_name="Distribution_Parameters", index=False)

        # NEW: Distribution Scores Sheet
        score_df.to_excel(
            writer, sheet_name="Distribution_Scores", index=False)

        # NEW: Summary Scorecard Sheet
        summary_card.to_excel(
            writer, sheet_name="Scorecard_Summary", index=False)

        # IDF and DDF tables for each distribution
        for name in dist_results.keys():
            idf_tables[name].to_excel(writer, sheet_name=f"IDF_{name}")
            ddf_tables[name].to_excel(writer, sheet_name=f"DDF_{name}")

    print(f"✓ Results successfully saved to: {OUTPUT_FILE}")

except Exception as e:
    print(f"✗ Error saving file: {e}")

# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------
print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)
print(f"📁 Input: Annual_Maximum_Rainfall.xlsx")
print(f"📊 Period: {years.min()}-{years.max()} ({len(data)} years)")
print(f"⏱ Durations: {len(durations_min)} durations")
print(f"🔄 Return periods: {len(T)} return periods")
print(f"🏆 Best distribution: {score_df.iloc[0]['Distribution']}")
print(f"   (Composite Score: {score_df.iloc[0]['Composite_Score']})")
print(f"💾 Output: {OUTPUT_FILE}")

print("\n" + "="*60)
print("RECOMMENDATION FOR IDF ANALYSIS")
print("="*60)
print(
    f"✓ Use the {score_df.iloc[0]['Distribution']} distribution for your IDF curves")
print(f"✓ {score_df.iloc[0]['Recommendation']}")
print("\n📊 NEW SHEETS ADDED:")
print("   • Distribution_Scores - Complete scoring matrix")
print("   • Scorecard_Summary - Quick reference summary")
print("="*60)
