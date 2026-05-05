#!/usr/bin/env python3
"""
Statistical Assumption Testing for Coverage Rate Data
Performs Shapiro-Wilk (normality) and Levene's (homogeneity of variance) tests
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load the data - change the path to wherever your CSV is
df = pd.read_csv('wsp_analysis_results.csv')

print("="*80)
print("STATISTICAL ASSUMPTION TESTING FOR ANOVA")
print("="*80)
print()

# Filter for drone data only (exclude backpack)
drone_data = df[~df['Height'].str.contains('Backpack', na=False)]

# ============================================================================
# TEST 1: NORMALITY (SHAPIRO-WILK TEST)
# ============================================================================
print("TEST 1: NORMALITY (Shapiro-Wilk Test)")
print("-" * 80)
print("Null Hypothesis: Data is normally distributed")
print("If p > 0.05: Data is probably normal ✓")
print("If p < 0.05: Data is probably NOT normal ✗")
print()

normality_results = []

for height in ['1.524m', '3.048m', '4.572m']:
    height_data = drone_data[drone_data['Height'] == height]['Coverage_Rate'].values
    statistic, p_value = stats.shapiro(height_data)
    
    normal_status = 'Normal' if p_value > 0.05 else 'NOT Normal'
    
    print(f'{height}:')
    print(f'  n = {len(height_data)}')
    print(f'  Mean = {np.mean(height_data):.4f}%')
    print(f'  SD = {np.std(height_data, ddof=1):.4f}%')
    print(f'  Shapiro-Wilk statistic = {statistic:.4f}')
    print(f'  p-value = {p_value:.6f} {normal_status}')
    print()

# Test overall normality
all_coverage = drone_data['Coverage_Rate'].values
stat, p = stats.shapiro(all_coverage)
print(f'ALL DRONE DATA COMBINED:')
print(f'  n = {len(all_coverage)}')
print(f'  Shapiro-Wilk statistic = {stat:.4f}')
print(f'  p-value = {p:.6f}')
print(f'  Status: {"Normal" if p > 0.05 else "NOT Normal"}')
print()

# ============================================================================
# TEST 2: HOMOGENEITY OF VARIANCE (LEVENE'S TEST)
# ============================================================================
print("="*80)
print("TEST 2: HOMOGENEITY OF VARIANCE (Levene's Test)")
print("-" * 80)
print("Null Hypothesis: All groups have equal variance")
print("If p > 0.05: Variances are probably equal")
print("If p < 0.05: Variances are probably UNEQUAL (Heteroscedasticity)")
print()

group_1524 = drone_data[drone_data['Height'] == '1.524m']['Coverage_Rate'].values
group_3048 = drone_data[drone_data['Height'] == '3.048m']['Coverage_Rate'].values
group_4572 = drone_data[drone_data['Height'] == '4.572m']['Coverage_Rate'].values

levene_stat, levene_p = stats.levene(group_1524, group_3048, group_4572)

print(f'Levene\'s Test (testing equality of variances across heights):')
print(f'  Test Statistic = {levene_stat:.4f}')
print(f'  p-value = {levene_p:.6f}')
print(f'  Status: {"Equal Variances" if levene_p > 0.05 else "UNEQUAL Variances (Heteroscedasticity)"}')
print()

print('Variances by Height:')
variance_results = []
for height, group in [('1.524m', group_1524), ('3.048m', group_3048), ('4.572m', group_4572)]:
    var = np.var(group, ddof=1)
    cv = (np.std(group, ddof=1) / np.mean(group)) * 100
    print(f'  {height}: Variance = {var:.6f}, CV = {cv:.2f}%')
    
    variance_results.append({
        'Height': height,
        'Variance': f'{var:.6f}',
        'CV_Percent': f'{cv:.2f}'
    })

print()

# ============================================================================
# SAVE RESULTS TO FILES
# ============================================================================

# Save normality results to CSV
normality_df = pd.DataFrame(normality_results)
normality_df.to_csv('Shapiro_Wilk_Normality_Test_Results.csv', index=False)
print("✓ Normality results saved to: Shapiro_Wilk_Normality_Test_Results.csv")

# Save variance results to CSV
variance_df = pd.DataFrame(variance_results)
variance_df.to_csv('Levene_Variance_Test_Results.csv', index=False)
print("✓ Variance results saved to: Levene_Variance_Test_Results.csv")

# Save all results to a comprehensive TXT file
with open('Statistical_Assumption_Test_Results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("STATISTICAL ASSUMPTION TESTING FOR ANOVA\n")
    f.write("="*80 + "\n\n")
    
    f.write("TEST 1: NORMALITY (Shapiro-Wilk Test)\n")
    f.write("-" * 80 + "\n")
    for result in normality_results:
        f.write(f"\n{result['Height']}:\n")
        f.write(f"  n = {result['n']}\n")
        f.write(f"  Mean = {result['Mean']}%\n")
        f.write(f"  SD = {result['SD']}%\n")
        f.write(f"  Shapiro-Wilk statistic = {result['Shapiro_Wilk_Statistic']}\n")
        f.write(f"  p-value = {result['p_value']}\n")
        f.write(f"  Normal: {result['Normal']}\n")
    
    f.write(f"\nALL DRONE DATA COMBINED:\n")
    f.write(f"  n = {len(all_coverage)}\n")
    f.write(f"  Shapiro-Wilk statistic = {stat:.4f}\n")
    f.write(f"  p-value = {p:.6f}\n")
    f.write(f"  Status: {'Normal' if p > 0.05 else 'NOT Normal'}\n\n")
    
    f.write("="*80 + "\n")
    f.write("TEST 2: HOMOGENEITY OF VARIANCE (Levene's Test)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Levene's Test Statistic = {levene_stat:.4f}\n")
    f.write(f"p-value = {levene_p:.6f}\n")
    f.write(f"Status: {'Equal Variances' if levene_p > 0.05 else 'UNEQUAL Variances'}\n\n")
    
    f.write("Variances by Height:\n")
    for result in variance_results:
        f.write(f"  {result['Height']}: Variance = {result['Variance']}, CV = {result['CV_Percent']}%\n")
    
print("✓ Complete results saved to: Statistical_Assumption_Test_Results.txt")
print()
print("All files saved successfully!")