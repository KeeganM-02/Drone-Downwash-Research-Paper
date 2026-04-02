# ==============================
# Drone Deposition Analysis Plots (Enhanced with Interaction Graphs + CSVs)
# ==============================
# Author: Keegan Miller
# Description:
#   Generates and saves mean, standard deviation, and box-and-whisker plots
#   of coverage rate, density, and droplet count
#   grouped by drone flight height and canopy position.
#   Includes automatic outlier handling, Height × Canopy Position interaction graphs,
#   and CSV exports of the numeric summaries.

import pandas as pd
import matplotlib.pyplot as plt
import os

# === Global Font Size Settings ===
plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.titlesize": 18,     # title size
    "axes.labelsize": 16,     # axis label size
    "xtick.labelsize": 14,    # x-axis tick labels
    "ytick.labelsize": 14,    # y-axis tick labels
    "legend.fontsize": 14     # legend text
})

# === Load data ===
df = pd.read_csv("wsp_analysis_results.csv")

# === Filter for drone heights only ===
df = df[df['Height'].isin(['5ft', '10ft', '15ft'])]

# === Ensure correct order for Height ===
height_order = ['5ft', '10ft', '15ft']
df['Height'] = pd.Categorical(df['Height'], categories=height_order, ordered=True)

# === Ensure correct order for Canopy Position ===
canopy_order = ['Top', 'Mid', 'Bot']  # adjust if you prefer Top→Bottom visual order
df['Canopy_Position'] = pd.Categorical(df['Canopy_Position'], categories=canopy_order, ordered=True)

# === Create output folders ===
output_dir = "plots_2"
os.makedirs(output_dir, exist_ok=True)

tables_dir = "tables"
os.makedirs(tables_dir, exist_ok=True)

# === Measurements to analyze ===
measurements = ['Coverage_Rate', 'Coverage_Density', 'Droplet_Count']

# === Dictionary for clean axis labels ===
pretty_labels = {
    'Coverage_Rate': 'Coverage Rate (%)',
    'Coverage_Density': 'Coverage Density',
    'Droplet_Count': 'Droplet Count',
    'Height': 'Flight Height',
    'Canopy_Position': 'Canopy Position'
}

# === Function to generate and save plots; returns grouped stats (mean/std/count) ===
def plot_stats(data, group_by, measurement):
    """
    Generates and saves mean, std, and box plots for a given measurement,
    grouped by a categorical variable (Height or Canopy_Position).
    Also creates a zoomed-in version excluding outliers (based on IQR).
    Returns a DataFrame with columns: [GroupBy, mean, std, count, Measurement, Grouping]
    """
    grouped = data.groupby(group_by, observed=True)[measurement].agg(['mean', 'std', 'count']).reset_index()

    y_label = pretty_labels[measurement]
    x_label = pretty_labels.get(group_by, group_by)

    # --- Mean Plot ---
    plt.figure(figsize=(8, 6))
    plt.bar(grouped[group_by], grouped['mean'], color='#0c5449') # WSU Primary Green
    plt.title(f'{pretty_labels[measurement]} Mean by {x_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{measurement}_Mean_by_{group_by}.png"), dpi=300)
    plt.close()

    # --- Standard Deviation Plot ---
    plt.figure(figsize=(8, 6))
    plt.bar(grouped[group_by], grouped['std'], color='#ffcc33')  # WSU Primary Yellow
    plt.title(f'{pretty_labels[measurement]} Standard Deviation by {x_label}')
    plt.xlabel(x_label)
    plt.ylabel(f'{y_label} Std Dev')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{measurement}_Std_by_{group_by}.png"), dpi=300)
    plt.close()

    # --- Box-and-Whisker Plot (Full Data) ---
    plt.figure(figsize=(8, 6))
    data.boxplot(column=measurement, by=group_by, grid=False, patch_artist=True,
                 boxprops=dict(facecolor='lightgreen', color='green'),
                 medianprops=dict(color='darkred', linewidth=1.5))
    plt.title(f'{pretty_labels[measurement]} by {x_label} (Full Range)')
    plt.suptitle('')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{measurement}_Box_by_{group_by}_Full.png"), dpi=300)
    plt.close()

    # --- Box-and-Whisker Plot (Zoomed / No Outliers) ---
    q1 = data[measurement].quantile(0.25)
    q3 = data[measurement].quantile(0.75)
    iqr = q3 - q1
    lower_lim = q1 - 1.5 * iqr
    upper_lim = q3 + 1.5 * iqr

    plt.figure(figsize=(8, 6))
    data.boxplot(column=measurement, by=group_by, grid=False, patch_artist=True,
                 boxprops=dict(facecolor='lightgreen', color='green'),
                 medianprops=dict(color='darkred', linewidth=1.5))
    plt.ylim(lower_lim, upper_lim)
    plt.title(f'{pretty_labels[measurement]} by {x_label} (Zoomed-In)')
    plt.suptitle('')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{measurement}_Box_by_{group_by}_Zoomed.png"), dpi=300)
    plt.close()

    # Return stats for CSV export
    grouped['Measurement'] = measurement
    grouped['Grouping'] = group_by
    return grouped

# === Function to create interaction (Height × Canopy Position) plots and return stats ===
def plot_interaction(data, measurement):
    """
    Generates interaction plots showing how Height and Canopy Position jointly
    influence a given measurement. Each line represents a canopy position across heights.
    Returns a DataFrame with columns: [Height, Canopy_Position, mean, std, count, Measurement, Grouping]
    """
    # For plotting (means)
    mean_for_plot = data.groupby(['Height', 'Canopy_Position'], observed=True)[measurement].mean().reset_index()

    plt.figure(figsize=(8, 6))
    for canopy in canopy_order:
        subset = mean_for_plot[mean_for_plot['Canopy_Position'] == canopy]
        plt.plot(subset['Height'], subset[measurement], marker='o', linewidth=2, label=canopy)

    plt.title(f'{pretty_labels[measurement]} by Height and Canopy Position')
    plt.xlabel(pretty_labels['Height'])
    plt.ylabel(pretty_labels[measurement])
    plt.legend(title='Canopy Position')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{measurement}_Interaction_Height_Canopy.png"), dpi=300)
    plt.close()

    # Full interaction stats for CSV
    stats = data.groupby(['Height', 'Canopy_Position'], observed=True)[measurement].agg(['mean', 'std', 'count']).reset_index()
    stats['Measurement'] = measurement
    stats['Grouping'] = 'Height×Canopy_Position'
    return stats

# === Generate and save all plots + collect stats ===
all_height_stats = []
all_canopy_stats = []
all_interaction_stats = []

for measurement in measurements:
    # By Height (with correct order)
    h_stats = plot_stats(df.sort_values('Height'), 'Height', measurement)
    all_height_stats.append(h_stats)

    # By Canopy Position
    c_stats = plot_stats(df, 'Canopy_Position', measurement)
    all_canopy_stats.append(c_stats)

    # Interaction Plot (Height × Canopy Position)
    i_stats = plot_interaction(df, measurement)
    all_interaction_stats.append(i_stats)

# === Concatenate and write CSVs ===
height_stats_df = pd.concat(all_height_stats, ignore_index=True)
canopy_stats_df = pd.concat(all_canopy_stats, ignore_index=True)
interaction_stats_df = pd.concat(all_interaction_stats, ignore_index=True)

# Reorder columns nicely
height_stats_df = height_stats_df[['Measurement', 'Grouping', 'Height', 'mean', 'std', 'count']]
canopy_stats_df = canopy_stats_df[['Measurement', 'Grouping', 'Canopy_Position', 'mean', 'std', 'count']]
interaction_stats_df = interaction_stats_df[['Measurement', 'Grouping', 'Height', 'Canopy_Position', 'mean', 'std', 'count']]

# Save individual CSVs
height_stats_df.to_csv(os.path.join(tables_dir, "summary_by_height.csv"), index=False)
canopy_stats_df.to_csv(os.path.join(tables_dir, "summary_by_canopy.csv"), index=False)
interaction_stats_df.to_csv(os.path.join(tables_dir, "summary_by_height_canopy.csv"), index=False)

# Save a combined CSV as well (union of all with consistent columns)
combined = pd.concat(
    [
        height_stats_df.assign(Canopy_Position=pd.NA),
        canopy_stats_df.assign(Height=pd.NA),
        interaction_stats_df
    ],
    ignore_index=True
)
combined = combined[['Measurement', 'Grouping', 'Height', 'Canopy_Position', 'mean', 'std', 'count']]
combined.to_csv(os.path.join(tables_dir, "summary_stats_all.csv"), index=False)

print(f"All plots (mean, std, box, zoom, and interaction) generated in '{output_dir}'.")
print(f"Numeric summaries saved to CSVs in '{tables_dir}'.")
