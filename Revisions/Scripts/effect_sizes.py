import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt

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

def analyze_drone_data(df):
    """Analyze drone data for effect sizes and confidence intervals"""
    
    # Filter out backpack data for drone analysis
    drone_data = df[~df['Height'].str.contains('Backpack', na=False)]
    
    # Get unique heights
    heights = ['1.524m', '3.048m', '4.572m']
    
    # Create lists to store results
    all_results = []
    summary_text = []
    
    summary_text.append("=== DRONE HEIGHT COMPARISONS ===\n")
    
    # For each pair of heights, calculate effect sizes
    comparisons = [
        ('1.524m', '4.572m'),  # Primary comparison from your Tukey test
        ('1.524m', '3.048m'),  # Additional comparisons
        ('3.048m', '4.572m')
    ]
    
    for height1, height2 in comparisons:
        summary_text.append(f"--- Comparing {height1} vs {height2} ---")
        
        # Get coverage rates for each height
        group1 = drone_data[drone_data['Height'] == height1]['Coverage_Rate'].values
        group2 = drone_data[drone_data['Height'] == height2]['Coverage_Rate'].values
        
        if len(group1) > 0 and len(group2) > 0:
            # Calculate statistics
            mean1, std1 = np.mean(group1), np.std(group1, ddof=1)
            mean2, std2 = np.mean(group2), np.std(group2, ddof=1)
            
            # Effect size
            cohens_d = calculate_cohens_d(group1, group2)
            effect_interpretation = interpret_effect_size(cohens_d)
            
            # Confidence interval
            mean_diff, ci = calculate_confidence_interval(group1, group2)
            
            # T-test for p-value
            t_stat, p_value = stats.ttest_ind(group1, group2)
            
            # Store results for CSV
            result = {
                'Comparison_Type': 'Height',
                'Group_1': height1,
                'Group_2': height2,
                'Group_1_Mean': mean1,
                'Group_1_SD': std1,
                'Group_1_n': len(group1),
                'Group_2_Mean': mean2,
                'Group_2_SD': std2,
                'Group_2_n': len(group2),
                'Mean_Difference': mean_diff,
                'CI_Lower_95': ci[0],
                'CI_Upper_95': ci[1],
                'Cohens_d': cohens_d,
                'Effect_Size_Interpretation': effect_interpretation,
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant_p005': 'Yes' if p_value < 0.05 else 'No'
            }
            all_results.append(result)
            
            # Add to summary text
            summary_text.append(f"  {height1}: Mean = {mean1:.3f}%, SD = {std1:.3f}% (n={len(group1)})")
            summary_text.append(f"  {height2}: Mean = {mean2:.3f}%, SD = {std2:.3f}% (n={len(group2)})")
            summary_text.append(f"  Mean difference: {mean_diff:.3f}%")
            summary_text.append(f"  95% CI: [{ci[0]:.3f}%, {ci[1]:.3f}%]")
            summary_text.append(f"  Cohen's d: {cohens_d:.3f} ({effect_interpretation} effect)")
            summary_text.append(f"  p-value: {p_value:.3f}")
            summary_text.append("")
    
    return all_results, summary_text

def analyze_canopy_positions(df):
    """Analyze canopy position differences"""
    
    # Filter drone data only
    drone_data = df[~df['Height'].str.contains('Backpack', na=False)]
    
    summary_text = []
    summary_text.append("=== CANOPY POSITION COMPARISONS (Drone Data) ===\n")
    
    # Compare Top vs Bottom (as mentioned in your Tukey test)
    top_data = drone_data[drone_data['Canopy_Position'] == 'Top']['Coverage_Rate'].values
    bot_data = drone_data[drone_data['Canopy_Position'] == 'Bot']['Coverage_Rate'].values
    mid_data = drone_data[drone_data['Canopy_Position'] == 'Mid']['Coverage_Rate'].values
    
    canopy_results = []
    
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
        canopy_results.append(result)
        
        summary_text.append("--- Top vs Bottom Canopy Position ---")
        summary_text.append(f"  Top: Mean = {mean_top:.3f}%, SD = {std_top:.3f}% (n={len(top_data)})")
        summary_text.append(f"  Bottom: Mean = {mean_bot:.3f}%, SD = {std_bot:.3f}% (n={len(bot_data)})")
        summary_text.append(f"  Mean difference (Top - Bottom): {mean_diff:.3f}%")
        summary_text.append(f"  95% CI: [{ci[0]:.3f}%, {ci[1]:.3f}%]")
        summary_text.append(f"  Cohen's d: {cohens_d:.3f} ({effect_interpretation} effect)")
        summary_text.append(f"  p-value: {p_value:.3f}")
        summary_text.append("")
    
    return canopy_results, summary_text

def main():
    """Main analysis function"""
    
    # Read the data
    df = pd.read_csv('wsp_analysis_results.csv')
    
    all_summary_text = []
    all_summary_text.append("Coverage Rate Statistical Analysis Summary")
    all_summary_text.append("=" * 50)
    all_summary_text.append("")
    all_summary_text.append(f"Data loaded successfully!")
    all_summary_text.append(f"Total records: {len(df)}")
    all_summary_text.append(f"Heights: {list(df['Height'].unique())}")
    all_summary_text.append(f"Canopy positions: {list(df['Canopy_Position'].unique())}")
    all_summary_text.append("")
    all_summary_text.append("="*60)
    all_summary_text.append("")
    
    # Analyze drone height comparisons
    height_results, height_summary = analyze_drone_data(df)
    all_summary_text.extend(height_summary)
    
    # Analyze canopy position comparisons
    canopy_results, canopy_summary = analyze_canopy_positions(df)
    all_summary_text.extend(canopy_summary)
    
    # Combine all results
    all_results = height_results + canopy_results
    
    # Create DataFrame for CSV output
    results_df = pd.DataFrame(all_results)
    
    # Save CSV file
    csv_filename = 'Coverage_Rate_Statistics.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Statistical results saved to: {csv_filename}")
    
    # Create summary for paper
    paper_summary = []
    paper_summary.append("=== SUMMARY FOR YOUR PAPER ===")
    paper_summary.append("")
    
    for result in height_results:
        if result['p_value'] < 0.05:
            paper_summary.append(f"{result['Group_1']} vs {result['Group_2']}:")
            paper_summary.append(f"  Mean difference: {result['Mean_Difference']:.2f}% "
                  f"(95% CI: {result['CI_Lower_95']:.2f}% to {result['CI_Upper_95']:.2f}%, "
                  f"Cohen's d = {result['Cohens_d']:.2f}, p = {result['p_value']:.3f})")
            paper_summary.append("")
    
    all_summary_text.extend(paper_summary)
    
    # Save summary text file
    txt_filename = 'Coverage_Rate_Stats_Summary.txt'
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(all_summary_text))
    print(f"Summary saved to: {txt_filename}")
    
    # Print summary to console
    print("\n".join(all_summary_text))

if __name__ == "__main__":
    main()