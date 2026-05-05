#!/usr/bin/env python3
"""
Statistical Analysis for Backpack Sprayer Trials
Calculates effect sizes, confidence intervals, and p-values for backpack data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups"""
    n1, n2 = len(group1), len(group2)
    
    # Calculate pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    
    # Calculate Cohen's d
    cohens_d = (np.mean(group2) - np.mean(group1)) / pooled_std
    return cohens_d

def interpret_effect_size(d):
    """Interpret Cohen's d effect size"""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"

def calculate_confidence_interval(group1, group2, confidence=0.95):
    """Calculate confidence interval for the difference between two means"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard error
    pooled_se = np.sqrt(var1/n1 + var2/n2)
    
    # Degrees of freedom (using Welch's approximation)
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # Critical t-value
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha/2, df)
    
    # Mean difference and margin of error
    mean_diff = mean2 - mean1
    margin_error = t_critical * pooled_se
    
    return mean_diff, (mean_diff - margin_error, mean_diff + margin_error)

def analyze_backpack_by_canopy(df, metric_name, metric_column):
    """Analyze backpack sprayer data by canopy position"""
    
    # Filter for backpack data only
    backpack_data = df[df['Height'].str.contains('Backpack', na=False)]
    
    summary_text = []
    summary_text.append(f"=== BACKPACK SPRAYER: {metric_name} BY CANOPY POSITION ===\n")
    
    # Get data by canopy position
    top_data = backpack_data[backpack_data['Canopy_Position'] == 'Top'][metric_column].dropna().values
    bot_data = backpack_data[backpack_data['Canopy_Position'] == 'Bot'][metric_column].dropna().values
    
    backpack_results = []
    
    # Top vs Bottom comparison
    if len(top_data) > 0 and len(bot_data) > 0:
        # Calculate statistics
        mean_top, std_top = np.mean(top_data), np.std(top_data, ddof=1)
        mean_bot, std_bot = np.mean(bot_data), np.std(bot_data, ddof=1)
        
        # Effect size
        cohens_d = calculate_cohens_d(bot_data, top_data)  # Bot vs Top
        effect_interpretation = interpret_effect_size(cohens_d)
        
        # Confidence interval
        mean_diff, ci = calculate_confidence_interval(bot_data, top_data)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(bot_data, top_data)
        
        # Store result
        result = {
            'Metric': metric_name,
            'Application_Method': 'Backpack Sprayer',
            'Comparison_Type': 'Canopy_Position',
            'Group_1': 'Bottom',
            'Group_2': 'Top',
            'Group_1_Mean': mean_bot,
            'Group_1_SD': std_bot,
            'Group_1_n': len(bot_data),
            'Group_2_Mean': mean_top,
            'Group_2_SD': std_top,
            'Group_2_n': len(top_data),
            'Mean_Difference': mean_diff,
            'CI_Lower_95': ci[0],
            'CI_Upper_95': ci[1],
            'Cohens_d': cohens_d,
            'Effect_Size_Interpretation': effect_interpretation,
            't_statistic': t_stat,
            'p_value': p_value,
            'Significant_p005': 'Yes' if p_value < 0.05 else 'No'
        }
        backpack_results.append(result)
        
        summary_text.append("--- Top vs Bottom Canopy Position ---")
        summary_text.append(f"  Top: Mean = {mean_top:.4f}, SD = {std_top:.4f} (n={len(top_data)})")
        summary_text.append(f"  Bottom: Mean = {mean_bot:.4f}, SD = {std_bot:.4f} (n={len(bot_data)})")
        summary_text.append(f"  Mean difference (Top - Bottom): {mean_diff:.4f}")
        summary_text.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        summary_text.append(f"  Cohen's d: {cohens_d:.3f} ({effect_interpretation} effect)")
        summary_text.append(f"  t-statistic: {t_stat:.3f}")
        summary_text.append(f"  p-value: {p_value:.3f}")
        summary_text.append("")
    
    return backpack_results, summary_text

def main():
    """Main analysis function for backpack data"""
    
    # Read the data
    df = pd.read_csv('wsp_analysis_results.csv')
    
    all_summary_text = []
    all_summary_text.append("="*70)
    all_summary_text.append("BACKPACK SPRAYER TRIAL STATISTICAL ANALYSIS")
    all_summary_text.append("="*70)
    all_summary_text.append("")
    all_summary_text.append(f"Data loaded successfully!")
    all_summary_text.append(f"Total records: {len(df)}")
    all_summary_text.append("")
    
    # Filter for backpack data
    backpack_data = df[df['Height'].str.contains('Backpack', na=False)]
    all_summary_text.append(f"Backpack sprayer records: {len(backpack_data)}")
    all_summary_text.append(f"Heights: {list(backpack_data['Height'].unique())}")
    all_summary_text.append(f"Canopy positions: {list(backpack_data['Canopy_Position'].unique())}")
    all_summary_text.append(f"Sample sizes: n = {backpack_data['Height'].value_counts().to_dict()}")
    all_summary_text.append("")
    all_summary_text.append("="*70)
    all_summary_text.append("")
    
    # Define metrics to analyze
    metrics = [
        ('Coverage Rate (%)', 'Coverage_Rate'),
        ('Coverage Density (droplets/mm²)', 'Coverage_Density'),
        ('Droplet Count', 'Droplet_Count')
    ]
    
    all_results = []
    
    # Analyze each metric
    for metric_name, metric_column in metrics:
        # Canopy position analysis for backpack
        canopy_results, canopy_summary = analyze_backpack_by_canopy(df, metric_name, metric_column)
        all_results.extend(canopy_results)
        all_summary_text.extend(canopy_summary)
        
        all_summary_text.append("="*70)
        all_summary_text.append("")
    
    # Create DataFrame for CSV output
    results_df = pd.DataFrame(all_results)
    
    # Save CSV file
    csv_filename = 'Backpack_Sprayer_Statistics.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Backpack sprayer statistical results saved to: {csv_filename}")
    
    # Create summary for paper
    paper_summary = []
    paper_summary.append("\n=== BACKPACK SPRAYER FINDINGS SUMMARY FOR YOUR PAPER ===")
    paper_summary.append("")
    
    # Group by metric for clarity
    for metric_name, _ in metrics:
        metric_results = results_df[results_df['Metric'] == metric_name]
        significant_results = metric_results[metric_results['Significant_p005'] == 'Yes']
        
        if len(significant_results) > 0:
            paper_summary.append(f"--- {metric_name} ---")
            for idx, result in significant_results.iterrows():
                paper_summary.append(
                    f"{result['Group_1']} vs {result['Group_2']} ({result['Comparison_Type']}):"
                )
                paper_summary.append(
                    f"  Mean difference: {result['Mean_Difference']:.4f} "
                    f"(95% CI: {result['CI_Lower_95']:.4f} to {result['CI_Upper_95']:.4f}, "
                    f"Cohen's d = {result['Cohens_d']:.2f}, p = {result['p_value']:.3f})"
                )
            paper_summary.append("")
        else:
            paper_summary.append(f"--- {metric_name} ---")
            paper_summary.append("No significant differences found between canopy positions.")
            paper_summary.append("")
    
    all_summary_text.extend(paper_summary)
    
    # Save summary text file
    txt_filename = 'Backpack_Sprayer_Stats_Summary.txt'
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(all_summary_text))
    print(f"Summary saved to: {txt_filename}")
    
    # Print summary to console
    print("\n".join(all_summary_text))

if __name__ == "__main__":
    main()