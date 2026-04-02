# ==============================
# Drone & Backpack Sprayer Deposition Statistical Analysis
# ==============================
# Description:
#   Runs one-way and two-way ANOVAs and Tukey HSD post-hoc tests
#   for each deposition measurement.
#   Includes one-way ANOVA for Backpack Sprayer Canopy Position.
#   Automatically saves results to CSV files.

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

# --- Load and filter data ---
df = pd.read_csv("wsp_analysis_results.csv")
df_drone = df[df['Height'].isin(['5ft', '10ft', '15ft'])].copy()
df_backpack = df[df['Height'].str.contains('Backpack Sprayer', case=False, na=False)].copy()


# --- Create output folder ---
output_dir = "stats_results"
os.makedirs(output_dir, exist_ok=True)

# --- Variables to test ---
measurements = ['Coverage_Rate', 'Coverage_Density', 'Droplet_Count']

# --- Prepare storage for summary tables ---
anova_summary = []
tukey_summary = []

# ==============================
# Drone Trials Analysis
# ==============================

for measurement in measurements:
    print(f"\n{'='*80}")
    print(f"DRONE ANALYSIS FOR: {measurement}")
    print(f"{'='*80}\n")

    # === One-way ANOVA: Height ===
    groups_height = [group[measurement].dropna().values for _, group in df_drone.groupby("Height")]
    f_h, p_h = stats.f_oneway(*groups_height)
    anova_summary.append([measurement, "Height (1-way)", f_h, p_h])
    print(f"Height effect on {measurement}: F = {f_h:.3f}, p = {p_h:.4f}")

    if p_h < 0.05:
        tukey_h = pairwise_tukeyhsd(endog=df_drone[measurement],
                                    groups=df_drone['Height'], alpha=0.05)
        print("\nTukey HSD (Height):")
        print(tukey_h)
        tukey_df = pd.DataFrame(data=tukey_h.summary().data[1:], columns=tukey_h.summary().data[0])
        tukey_df['Measurement'] = measurement
        tukey_df['Factor'] = 'Height'
        tukey_summary.append(tukey_df)

    # === One-way ANOVA: Canopy Position ===
    groups_canopy = [group[measurement].dropna().values for _, group in df_drone.groupby("Canopy_Position")]
    f_c, p_c = stats.f_oneway(*groups_canopy)
    anova_summary.append([measurement, "Canopy_Position (1-way)", f_c, p_c])
    print(f"\nCanopy Position effect on {measurement}: F = {f_c:.3f}, p = {p_c:.4f}")

    if p_c < 0.05:
        tukey_c = pairwise_tukeyhsd(endog=df_drone[measurement],
                                    groups=df_drone['Canopy_Position'], alpha=0.05)
        print("\nTukey HSD (Canopy Position):")
        print(tukey_c)
        tukey_df = pd.DataFrame(data=tukey_c.summary().data[1:], columns=tukey_c.summary().data[0])
        tukey_df['Measurement'] = measurement
        tukey_df['Factor'] = 'Canopy_Position'
        tukey_summary.append(tukey_df)

    # === Two-way ANOVA ===
    model = ols(f'{measurement} ~ C(Height) + C(Canopy_Position) + C(Height):C(Canopy_Position)', data=df_drone).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\nTwo-way ANOVA ({measurement} ~ Height × Canopy_Position):\n")
    print(anova_table)
    
    for factor in anova_table.index:
        anova_summary.append([
            measurement,
            factor,
            anova_table.loc[factor, 'F'],
            anova_table.loc[factor, 'PR(>F)']
        ])

    print("\n" + "-"*80 + "\n")

# ==============================
# Backpack Sprayer Analysis
# ==============================

for measurement in measurements:
    print(f"\n{'='*80}")
    print(f"BACKPACK SPRAYER: Canopy Effect on {measurement}")
    print(f"{'='*80}\n")

    groups_canopy_bp = [group[measurement].dropna().values for _, group in df_backpack.groupby("Canopy_Position")]
    
    # Skip if not enough groups or missing data
    if len(groups_canopy_bp) < 2 or any(len(g) == 0 for g in groups_canopy_bp):
        print(f"Not enough valid data for {measurement} — skipping.\n")
        continue

    # One-way ANOVA for Backpack Sprayer canopy positions
    f_bp, p_bp = stats.f_oneway(*groups_canopy_bp)
    print(f"One-way ANOVA (Backpack - Canopy): F = {f_bp:.3f}, p = {p_bp:.4f}")
    anova_summary.append([measurement, "Canopy_Position (Backpack)", f_bp, p_bp])

    # Tukey HSD post-hoc if significant
    if p_bp < 0.05:
        tukey_bp = pairwise_tukeyhsd(endog=df_backpack[measurement],
                                     groups=df_backpack['Canopy_Position'], alpha=0.05)
        print("\nTukey HSD (Backpack Sprayer - Canopy Position):")
        print(tukey_bp)
        tukey_df = pd.DataFrame(data=tukey_bp.summary().data[1:], columns=tukey_bp.summary().data[0])
        tukey_df['Measurement'] = measurement
        tukey_df['Factor'] = 'Canopy_Position (Backpack)'
        tukey_summary.append(tukey_df)

    print("\n" + "-"*80 + "\n")

# ==============================
# Export All Results
# ==============================

anova_df = pd.DataFrame(anova_summary, columns=["Measurement", "Factor", "F", "p"])
anova_df['Significant'] = anova_df['p'].apply(lambda x: '*' if x < 0.05 else '')
anova_df.to_csv(os.path.join(output_dir, "ANOVA_Results2.csv"), index=False)

if tukey_summary:
    tukey_all = pd.concat(tukey_summary, ignore_index=True)
    tukey_all.to_csv(os.path.join(output_dir, "TukeyHSD_Results2.csv"), index=False)

print(f"\nAll statistical results saved in '{output_dir}' folder.")
