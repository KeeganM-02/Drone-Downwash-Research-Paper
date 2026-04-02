import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_wsp_results(csv_path, output_dir="."):
    # Load dataset
    df = pd.read_csv(csv_path)

    # --- Aggregated statistics ---
    # Mean deposition by canopy position
    mean_dep_canopy = df.groupby("Canopy_Position")["Deposition"].mean()

    # Mean deposition by height
    mean_dep_height = df.groupby("Height")["Deposition"].mean()

    # Variance of droplet count by height
    var_droplet_height = df.groupby("Height")["Droplet_Count"].var()

    # Mean coverage by canopy position
    mean_coverage_canopy = df.groupby("Canopy_Position")["Coverage_Rate"].mean()

    # Mean coverage density by height
    mean_density_height = df.groupby("Height")["Coverage_Density"].mean()

    # Combine summaries into a single CSV
    summary = pd.concat([
        mean_dep_canopy.rename("Mean_Deposition_by_Canopy"),
        mean_dep_height.rename("Mean_Deposition_by_Height"),
        var_droplet_height.rename("Variance_Droplet_Count_by_Height"),
        mean_coverage_canopy.rename("Mean_Coverage_by_Canopy"),
        mean_density_height.rename("Mean_Coverage_Density_by_Height")
    ], axis=1)

    summary_csv = f"{output_dir}/wsp_summary_stats.csv"
    summary.to_csv(summary_csv)
    print(f"Summary statistics saved to {summary_csv}")

    # --- Plots ---
    sns.set(style="whitegrid")

    # Mean deposition by canopy position
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="Canopy_Position", y="Deposition", ci="sd")
    plt.title("Mean Deposition by Canopy Position")
    plt.ylabel("Deposition (mm²)")
    plt.savefig(f"{output_dir}/mean_deposition_by_canopy.png", dpi=300)
    plt.close()

    # Mean deposition by height
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="Height", y="Deposition", ci="sd")
    plt.title("Mean Deposition by Height")
    plt.ylabel("Deposition (mm²)")
    plt.savefig(f"{output_dir}/mean_deposition_by_height.png", dpi=300)
    plt.close()

    # Droplet count variance by height
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="Height", y="Droplet_Count")
    plt.title("Droplet Count Distribution by Height")
    plt.ylabel("Droplet Count")
    plt.savefig(f"{output_dir}/droplet_count_by_height.png", dpi=300)
    plt.close()

    # Mean coverage rate by canopy
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="Canopy_Position", y="Coverage_Rate", ci="sd")
    plt.title("Coverage Rate by Canopy Position")
    plt.ylabel("Coverage (%)")
    plt.savefig(f"{output_dir}/coverage_rate_by_canopy.png", dpi=300)
    plt.close()

    # Mean coverage density by height
    plt.figure(figsize=(6,4))
    sns.barplot(data=df, x="Height", y="Coverage_Density", ci="sd")
    plt.title("Coverage Density by Height")
    plt.ylabel("Coverage Density (droplets/mm²)")
    plt.savefig(f"{output_dir}/coverage_density_by_height.png", dpi=300)
    plt.close()

    print("Plots saved to output directory.")


if __name__ == "__main__":
    csv_path = r"C:\Users\keega\Desktop\WSP_Images\wsp_analysis_results.csv"
    analyze_wsp_results(csv_path, output_dir=r"C:\Users\keega\Desktop\WSP_Images")
