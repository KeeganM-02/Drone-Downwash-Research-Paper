# import pandas as pd

# # Load data
# df = pd.read_csv("wsp_analysis_results.csv")

# # Group by height and plant (each trial), calculate variance
# coverage_rate_variance = df.groupby(['Height', 'Plant'])['Coverage_Rate'].var()
# coverage_density_variance = df.groupby(['Height', 'Plant'])['Coverage_Density'].var()
# droplet_count_variance = df.groupby(['Height', 'Plant'])['Droplet_Count'].var()

# print("Coverage Rate Variance")
# print(coverage_rate_variance)
# print("Coverage Density Variance")
# print(coverage_density_variance)
# print("Droplet Count Variance")
# print(droplet_count_variance)

import pandas as pd

# Load your CSV
df = pd.read_csv("wsp_analysis_results.csv")

# Function to calculate CV for any column
def calc_cv(df, column):
    stats = df.groupby(['Height', 'Plant'])[column].agg(['mean', 'std'])
    stats['CV (%)'] = (stats['std'] / stats['mean']) * 100
    return stats[['CV (%)']]

# Compute CV for each metric
cv_coverage_rate = calc_cv(df, 'Coverage_Rate')
cv_coverage_density = calc_cv(df, 'Coverage_Density')
cv_droplet_count = calc_cv(df, 'Droplet_Count')

print("Coverage Rate CV:\n", cv_coverage_rate)
print("\nCoverage Density CV:\n", cv_coverage_density)
print("\nDroplet Count CV:\n", cv_droplet_count)

coveragerate_by_trial = (
    df.groupby('Height')['Coverage_Rate']
      .agg(lambda x: (x.std() / x.mean()) * 100)
      .reset_index(name='CV (%)')
)
coveragedensity_by_trial = (
    df.groupby('Height')['Coverage_Density']
      .agg(lambda x: (x.std() / x.mean()) * 100)
      .reset_index(name='CV (%)')
)
dropletcount_by_trial = (
    df.groupby('Height')['Droplet_Count']
      .agg(lambda x: (x.std() / x.mean()) * 100)
      .reset_index(name='CV (%)')
)

print("Coverage Rate")
print(coveragerate_by_trial)
print("Coverage Density")
print(coveragedensity_by_trial)
print("Droplet Count")
print(dropletcount_by_trial)
