# ============================================================
# COMPLETE IDF ANALYSIS (MULTI-DISTRIBUTION)
# WITH FIXED GOODNESS-OF-FIT TESTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm, lognorm, gumbel_r, genextreme, skew, kstest, chi2, anderson
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_PATH = r"E:\BEST Pak\Meteorological Data\Frequency_Analysis"
INPUT_FILE = os.path.join(BASE_PATH, "Rainfall_IDF.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "IDF_All_Distributions.xlsx")
PLOT_FOLDER = os.path.join(BASE_PATH, "IDF_Plots")

# Create plot folder if it doesn't exist
os.makedirs(PLOT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# READ DATA
# ------------------------------------------------------------
df = pd.read_excel(INPUT_FILE)
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Observed_Rainfall"] = pd.to_numeric(
    df["Observed_Rainfall"], errors="coerce")
df = df.dropna(subset=["Date", "Observed_Rainfall"])
df = df[df["Observed_Rainfall"] > 0]
df = df.set_index("Date")
print("✔ Data loaded")

# ------------------------------------------------------------
# ANNUAL MAXIMUM SERIES
# ------------------------------------------------------------
ams = df.resample("YE")["Observed_Rainfall"].max().dropna()
data = ams.values
sorted_data = np.sort(data)
n = len(data)

print("✔ Annual maxima computed")
print(f"   Number of years: {n}")

# ------------------------------------------------------------
# BASIC STATISTICS
# ------------------------------------------------------------
mean_val = np.mean(data)
median_val = np.median(data)
std_val = np.std(data, ddof=1)
var_val = np.var(data, ddof=1)
skewness_val = skew(data, bias=False)
kurtosis_val = stats.kurtosis(data, bias=False)
cv_val = std_val / mean_val
min_val = np.min(data)
max_val = np.max(data)
range_val = max_val - min_val
q1_val = np.percentile(data, 25)
q3_val = np.percentile(data, 75)
iqr_val = q3_val - q1_val

print("\n" + "="*60)
print("BASIC STATISTICS OF ANNUAL MAXIMA")
print("="*60)
print(f"Mean:                      {mean_val:.2f} mm")
print(f"Median:                    {median_val:.2f} mm")
print(f"Standard Deviation:        {std_val:.2f} mm")
print(f"Coefficient of Variation:  {cv_val:.3f}")
print(f"Skewness:                  {skewness_val:.3f}")
print(f"Kurtosis:                  {kurtosis_val:.3f}")
print(f"Minimum Value:             {min_val:.2f} mm")
print(f"Maximum Value:             {max_val:.2f} mm")
print(f"Number of years:           {n}")

# ------------------------------------------------------------
# RETURN PERIODS
# ------------------------------------------------------------
T = np.array([2, 3, 4, 5, 10, 20, 50, 100, 500, 1000])
P = 1 - 1 / T

print(f"\nReturn periods considered: {T}")

# ------------------------------------------------------------
# DISTRIBUTION FITTING
# ------------------------------------------------------------
dist_params = {}

# --- NORMAL ---
mu_norm, sigma_norm = norm.fit(data)
dist_params["Normal"] = {
    "mu": mu_norm,
    "sigma": sigma_norm,
    "params_text": f"μ = {mu_norm:.2f}, σ = {sigma_norm:.2f}"
}

# --- LOGNORMAL (FIXED) ---
shape_ln, loc_ln, scale_ln = lognorm.fit(data, floc=0)
dist_params["LogNormal"] = {
    "shape": shape_ln,
    "loc": loc_ln,
    "scale": scale_ln,
    "params_text": f"Shape = {shape_ln:.3f}, Scale = {scale_ln:.2f}"
}

# --- GUMBEL ---
loc_gumb, scale_gumb = gumbel_r.fit(data)
dist_params["Gumbel"] = {
    "loc": loc_gumb,
    "scale": scale_gumb,
    "params_text": f"Location = {loc_gumb:.2f}, Scale = {scale_gumb:.2f}"
}

# --- GEV ---
c_gev, loc_gev, scale_gev = genextreme.fit(data)
dist_params["GEV"] = {
    "c": c_gev,
    "loc": loc_gev,
    "scale": scale_gev,
    "params_text": f"Shape(c) = {c_gev:.3f}, Location = {loc_gev:.2f}, Scale = {scale_gev:.2f}"
}

# --- LP3 ---
log_data = np.log10(data)
mean_lp3 = log_data.mean()
std_lp3 = log_data.std(ddof=1)
cs_lp3 = skew(log_data, bias=False)
dist_params["LP3"] = {
    "mean_log": mean_lp3,
    "std_log": std_lp3,
    "skew_log": cs_lp3,
    "params_text": f"Mean(log) = {mean_lp3:.3f}, Std(log) = {std_lp3:.3f}, Skew(log) = {cs_lp3:.3f}"
}

# ------------------------------------------------------------
# FIXED GOODNESS-OF-FIT TESTS
# ------------------------------------------------------------


def kolmogorov_smirnov_test_fixed(data, dist_name, dist, params):
    """Fixed Kolmogorov-Smirnov test"""
    try:
        if dist_name == "LP3":
            return np.nan, np.nan, "N/A", "LP3 not implemented in scipy.stats"

        # For LogNormal, ensure proper parameter format
        if dist_name == "LogNormal":
            # LogNormal in scipy uses (shape, loc, scale)
            ks_stat, p_value = kstest(
                data, lambda x: dist.cdf(x, *params), args=())
        else:
            # For other distributions
            ks_stat, p_value = kstest(data, dist.cdf, args=params)

        alpha = 0.05
        decision = "Accepted" if p_value > alpha else "Rejected"
        reason = f"p-value = {p_value:.4f}"

        return ks_stat, p_value, decision, reason
    except Exception as e:
        return np.nan, np.nan, "Error", str(e)


def chi_square_test_fixed(data, dist_name, dist, params):
    """Fixed Chi-Square test with adaptive binning"""
    try:
        if dist_name == "LP3":
            return np.nan, np.nan, np.nan, "N/A", "LP3 not implemented"

        # Use more bins for better distribution
        n_bins = max(5, min(int(np.sqrt(len(data))), 10))

        # Create bins with equal probability
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(data, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate observed frequencies
        observed, _ = np.histogram(data, bins=bin_edges)

        # Calculate expected frequencies
        expected = []
        for i in range(len(bin_edges) - 1):
            if dist_name == "LogNormal":
                prob = dist.cdf(bin_edges[i+1], *params) - \
                    dist.cdf(bin_edges[i], *params)
            else:
                prob = dist.cdf(bin_edges[i+1], *params) - \
                    dist.cdf(bin_edges[i], *params)
            expected.append(prob * len(data))

        expected = np.array(expected)

        # Remove bins with expected frequency < 5
        mask = expected >= 5
        observed = observed[mask]
        expected = expected[mask]

        if len(observed) < 2:
            return np.nan, np.nan, np.nan, "Insufficient data", "Too few bins after filtering"

        # Calculate chi-square statistic
        chi2_stat = np.sum((observed - expected) ** 2 / expected)

        # Degrees of freedom
        dof = len(observed) - 1 - len(params)

        if dof <= 0:
            return chi2_stat, np.nan, dof, "Invalid DOF", f"DOF = {dof} (too few bins)"

        # Calculate p-value
        p_value = 1 - chi2.cdf(chi2_stat, dof)

        alpha = 0.05
        decision = "Accepted" if p_value > alpha else "Rejected"
        reason = f"χ² = {chi2_stat:.3f}, df = {dof}, p = {p_value:.4f}"

        return chi2_stat, p_value, dof, decision, reason
    except Exception as e:
        return np.nan, np.nan, np.nan, "Error", str(e)


def anderson_darling_test_fixed(data, dist_name):
    """Fixed Anderson-Darling test"""
    try:
        if dist_name == "LP3":
            return np.nan, np.nan, "N/A", "LP3 not implemented in AD test"

        # Map distribution names to scipy's anderson format
        dist_map = {
            "Normal": 'norm',
            "LogNormal": 'lognorm',
            "Gumbel": 'gumbel_r',
            "GEV": 'extreme1'
        }

        if dist_name not in dist_map:
            return np.nan, np.nan, "N/A", "Distribution not available"

        result = anderson(data, dist=dist_map[dist_name])
        ad_stat = result.statistic
        critical_values = result.critical_values

        # Get critical value at 95% confidence
        cv_95 = critical_values[2]

        decision = "Accepted" if ad_stat < cv_95 else "Rejected"
        reason = f"A-D = {ad_stat:.3f} < CV(95%) = {cv_95:.3f}" if ad_stat < cv_95 else f"A-D = {ad_stat:.3f} > CV(95%) = {cv_95:.3f}"

        return ad_stat, cv_95, decision, reason
    except Exception as e:
        return np.nan, np.nan, "Error", str(e)


# ------------------------------------------------------------
# CALCULATE 24-HOUR DEPTHS
# ------------------------------------------------------------
dist_results = {}
dist_results["Normal"] = norm.ppf(P, mu_norm, sigma_norm)
dist_results["LogNormal"] = lognorm.ppf(P, shape_ln, loc_ln, scale_ln)
dist_results["Gumbel"] = gumbel_r.ppf(P, loc_gumb, scale_gumb)
dist_results["GEV"] = genextreme.ppf(P, c_gev, loc_gev, scale_gev)

# LP3 calculation
z = norm.ppf(P)
lp3 = mean_lp3 + z * std_lp3 * (1 + (cs_lp3 * z / 6))
dist_results["LP3"] = 10 ** lp3

# ------------------------------------------------------------
# DURATIONS
# ------------------------------------------------------------
durations_min = np.array([5, 15, 30, 60, 120, 360, 720, 1440, 2880, 4320])
durations_hr = durations_min / 60
duration_labels = ['5min', '15min', '30min', '1hr',
                   '2hrs', '6hrs', '12hrs', '24hrs', '48hrs', '72hrs']

# ------------------------------------------------------------
# GENERATE IDF
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


idf_tables = {}
for name, values in dist_results.items():
    idf_tables[name] = generate_idf(values)

# ------------------------------------------------------------
# PERFORM ALL TESTS
# ------------------------------------------------------------
print("\n" + "="*80)
print("GOODNESS-OF-FIT TESTS RESULTS (FIXED)")
print("="*80)

test_results = []

for dist_name in ["Normal", "LogNormal", "Gumbel", "GEV", "LP3"]:
    print(f"\n{'-'*40}")
    print(f"{dist_name} DISTRIBUTION")
    print(f"{'-'*40}")
    print(f"Parameters: {dist_params[dist_name]['params_text']}")

    if dist_name == "Normal":
        dist = norm
        params = (mu_norm, sigma_norm)
    elif dist_name == "LogNormal":
        dist = lognorm
        params = (shape_ln, loc_ln, scale_ln)
    elif dist_name == "Gumbel":
        dist = gumbel_r
        params = (loc_gumb, scale_gumb)
    elif dist_name == "GEV":
        dist = genextreme
        params = (c_gev, loc_gev, scale_gev)
    else:  # LP3
        dist = None
        params = None

    # Kolmogorov-Smirnov Test
    ks_stat, ks_p, ks_decision, ks_reason = kolmogorov_smirnov_test_fixed(
        data, dist_name, dist, params)
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  KS Statistic: {ks_stat:.4f}" if not np.isnan(
        ks_stat) else f"  KS Statistic: {ks_reason}")
    print(f"  P-Value: {ks_p:.4f}" if not np.isnan(ks_p) else "  P-Value: N/A")
    print(f"  Decision: {ks_decision}")

    # Chi-Square Test
    chi2_stat, chi2_p, dof, chi2_decision, chi2_reason = chi_square_test_fixed(
        data, dist_name, dist, params)
    print(f"\nChi-Square Test:")
    print(f"  Chi-Square Statistic: {chi2_stat:.3f}" if not np.isnan(
        chi2_stat) else f"  Chi-Square: {chi2_reason}")
    print(f"  Degrees of Freedom: {dof}" if not np.isnan(
        dof) else "  DOF: N/A")
    print(f"  P-Value: {chi2_p:.4f}" if not np.isnan(chi2_p)
          else "  P-Value: N/A")
    print(f"  Decision: {chi2_decision}")

    # Anderson-Darling Test (only for non-LP3)
    if dist_name != "LP3":
        ad_stat, ad_cv, ad_decision, ad_reason = anderson_darling_test_fixed(
            data, dist_name)
        print(f"\nAnderson-Darling Test:")
        print(
            f"  A-D Statistic: {ad_stat:.3f}" if not np.isnan(ad_stat) else f"  A-D: {ad_reason}")
        print(f"  Critical Value (95%): {ad_cv:.3f}" if not np.isnan(
            ad_cv) else "  CV: N/A")
        print(f"  Decision: {ad_decision}")
    else:
        ad_stat, ad_cv, ad_decision, ad_reason = np.nan, np.nan, "N/A", "Not applicable"

    # Store results
    test_results.append({
        "Distribution": dist_name,
        "Parameters": dist_params[dist_name]['params_text'],
        "KS_Statistic": ks_stat,
        "KS_PValue": ks_p,
        "KS_Decision": ks_decision,
        "Chi2_Statistic": chi2_stat,
        "Chi2_PValue": chi2_p,
        "Chi2_DOF": dof,
        "Chi2_Decision": chi2_decision,
        "AD_Statistic": ad_stat,
        "AD_Critical_95": ad_cv,
        "AD_Decision": ad_decision
    })

# ------------------------------------------------------------
# SELECT BEST DISTRIBUTION
# ------------------------------------------------------------


def select_best_distribution_fixed(test_results):
    """Select best distribution based on test results"""
    scores = {}

    for result in test_results:
        if result["Distribution"] == "LP3":
            continue

        score = 0

        # Score based on KS test
        if not np.isnan(result["KS_Statistic"]):
            score += (1 - min(result["KS_Statistic"], 1)) * 20
            if result["KS_Decision"] == "Accepted":
                score += 10

        # Score based on Chi-Square test
        if not np.isnan(result["Chi2_Statistic"]) and result["Chi2_Decision"] == "Accepted":
            score += 10

        # Score based on AD test
        if not np.isnan(result["AD_Statistic"]) and result["AD_Decision"] == "Accepted":
            score += 10

        scores[result["Distribution"]] = score

    if scores:
        best_dist = max(scores, key=scores.get)
        best_result = next(
            r for r in test_results if r["Distribution"] == best_dist)

        reasons = []
        if best_result["KS_Decision"] == "Accepted":
            reasons.append(
                f"Passed K-S test (p={best_result['KS_PValue']:.4f})")
        if best_result["Chi2_Decision"] == "Accepted":
            reasons.append(
                f"Passed Chi-Square test (p={best_result['Chi2_PValue']:.4f})")
        if not np.isnan(best_result["AD_Statistic"]) and best_result["AD_Decision"] == "Accepted":
            reasons.append(
                f"Passed A-D test (stat={best_result['AD_Statistic']:.3f})")

        if not reasons:
            reasons.append("Highest overall score in goodness-of-fit metrics")

        return best_dist, scores, reasons

    return "GEV", {"GEV": 50}, ["GEV is standard for extreme value analysis"]


best_distribution, scores, selection_reasons = select_best_distribution_fixed(
    test_results)

print("\n" + "="*80)
print("BEST DISTRIBUTION SELECTION")
print("="*80)
print(f"\n✓ Selected Best Distribution: {best_distribution}")
print(f"\nScores:")
for dist, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {dist:10s}: {score:.1f} points")
print(f"\nReasons for Selection:")
for reason in selection_reasons:
    print(f"  ✓ {reason}")

# ------------------------------------------------------------
# SAVE TO EXCEL
# ------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Basic Statistics
    stats_df = pd.DataFrame({
        "Statistic": ["Mean (mm)", "Median (mm)", "Std Dev (mm)", "CV", "Skewness",
                      "Kurtosis", "Minimum (mm)", "Maximum (mm)", "Number of Years"],
        "Value": [f"{mean_val:.2f}", f"{median_val:.2f}", f"{std_val:.2f}",
                  f"{cv_val:.4f}", f"{skewness_val:.4f}", f"{kurtosis_val:.4f}",
                  f"{min_val:.2f}", f"{max_val:.2f}", n]
    })
    stats_df.to_excel(writer, sheet_name="Basic_Statistics", index=False)

    # Test Results
    gof_df = pd.DataFrame(test_results)
    gof_df.to_excel(writer, sheet_name="Goodness_of_Fit", index=False)

    # Best Distribution
    pd.DataFrame({"Best_Distribution": [best_distribution],
                  "Reasons": [", ".join(selection_reasons)]}).to_excel(writer, sheet_name="Best_Distribution", index=False)

    # Scores
    pd.DataFrame(list(scores.items()), columns=["Distribution", "Score"]).to_excel(
        writer, sheet_name="Scores", index=False)

    # Annual Maxima
    pd.DataFrame({'Year': ams.index.year, 'Rainfall_mm': data}).to_excel(
        writer, sheet_name="Annual_Maxima", index=False)

    # IDF Tables
    for name, table in idf_tables.items():
        table.to_excel(writer, sheet_name=f"IDF_{name}")

print(f"\n✓ Results saved to: {OUTPUT_FILE}")

# ------------------------------------------------------------
# SIMPLE PLOT
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.hist(data, bins=15, density=True, alpha=0.5,
         color='lightblue', edgecolor='black', label='Observed')
x = np.linspace(data.min(), data.max(), 200)
plt.plot(x, norm.pdf(x, mu_norm, sigma_norm),
         'b-', label='Normal', linewidth=2)
plt.plot(x, lognorm.pdf(x, shape_ln, 0, scale_ln),
         'g-', label='LogNormal', linewidth=2)
plt.plot(x, gumbel_r.pdf(x, loc_gumb, scale_gumb),
         'r-', label='Gumbel', linewidth=2)
plt.plot(x, genextreme.pdf(x, c_gev, loc_gev, scale_gev),
         'm-', label='GEV', linewidth=2)
plt.xlabel('Annual Maximum Rainfall (mm)')
plt.ylabel('Probability Density')
plt.title(f'Distribution Fitting - Best: {best_distribution}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_FOLDER, "Distribution_Fitting.png"), dpi=300)
plt.close()

print(f"\n✓ Plot saved to: {PLOT_FOLDER}")
print("\n" + "="*80)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("="*80)
